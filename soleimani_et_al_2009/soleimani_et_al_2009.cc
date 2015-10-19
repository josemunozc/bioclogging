/*
 * Program to reproduce the results found in
 * Soleimani_et_al_2009 "Modelling of biological
 * in unsaturated porous media"
 *
 * It solves the coupled system in 1D and 2D:
 * -- Moisture movement (Darcy's equation)
 * -- Diffusion-Advection of organic matter
*/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h> 
#include <deal.II/base/multithread_info.h>  
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/parameter_handler.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/lac/sparse_matrix.h>   
#include <deal.II/lac/solver_bicgstab.h> 
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h> 
//#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream> 
#include <vector>
#include <string>

#include "/home/zerpiko/libraries/SurfaceCoefficients.h"
#include "/home/zerpiko/libraries/AnalyticSolution.h"
#include "/home/zerpiko/libraries/BoundaryConditions.h"
#include "/home/zerpiko/libraries/MaterialData.h"
#include "/home/zerpiko/libraries/data_tools.h"


void hydraulic_properties(
		double pressure_head,
		double &specific_moisture_capacity,
		double &hydraulic_conductivity,
		double &moisture_content,
		std::string hydraulic_properties)
{
	if (hydraulic_properties.compare("haverkamp_et_al_1977")==0)
	{
		double alpha=1.611E6;
		double beta =3.96;
		double moisture_content_saturation=0.287;
		double moisture_content_residual=0.075;
		double hydraulic_conductivity_saturated=0.00944; // (cm/s)
		double gamma=4.74;
		double A=1.175E6;

		specific_moisture_capacity
		=-1.*alpha*(moisture_content_saturation-moisture_content_residual)
		*beta*pressure_head*pow(fabs(pressure_head),beta-2)
		/pow(alpha+pow(fabs(pressure_head),beta),2);

		hydraulic_conductivity
		=hydraulic_conductivity_saturated*A/(A+pow(fabs(pressure_head),gamma));

		moisture_content
		=(alpha*(moisture_content_saturation-moisture_content_residual)/
				(alpha+pow(fabs(pressure_head),beta)))+moisture_content_residual;
	}
	else if (hydraulic_properties.compare("van_genuchten_1980")==0)
	{
		double alpha=0.0335;
		double moisture_content_saturation=0.368;
		double moisture_content_residual=0.102;
		double n=2;
		double m=0.5;
		double hydraulic_conductivity_saturated=0.00922*1.01; // (cm/s)

		specific_moisture_capacity
		=-1.*alpha*m*n*(moisture_content_saturation-moisture_content_residual)*
		pow(alpha*fabs(pressure_head),n-1.)*pow(1.+pow(alpha*fabs(pressure_head),n),-1.*m-1.)*
		pressure_head/fabs(pressure_head);

		hydraulic_conductivity
		=hydraulic_conductivity_saturated*
		pow(1.-pow(alpha*fabs(pressure_head),n-1.)*pow(1.+pow(alpha*fabs(pressure_head),n),-m),2.)
		/pow(1.+pow(alpha*fabs(pressure_head),n),m/2.);

		moisture_content
		=((moisture_content_saturation-moisture_content_residual)/
				pow(1.+pow(alpha*fabs(pressure_head),n),m))+moisture_content_residual;
	}
	else
	{
		std::cout << "Equations for \"" << hydraulic_properties
				<< "\" are not implemented. Error.\n";
		throw -1;
	}
}

namespace TRL
{
  using namespace dealii;
#include "/home/zerpiko/libraries/InitialValue.h"
#include "parameters.h"

  template <int dim>
  class Heat_Pipe
  {
  public:
    Heat_Pipe(int argc, char *argv[]);
    ~Heat_Pipe();
    void run();

  private:
    void read_grid_pressure();
    void setup_system_pressure();
    void initial_condition();
    void assemble_system_pressure();
    void solve_pressure();
    void output_results() const;

    void calculate_mass_balance_ratio();

    Triangulation<dim>   triangulation_pressure;
    DoFHandler<dim>      dof_handler_pressure;
    FE_Q<dim>            fe_pressure;
    
    ConstraintMatrix     hanging_node_constraints_pressure;
    
    SparsityPattern      sparsity_pattern_pressure;

    SparseMatrix<double> system_matrix_pressure;
    SparseMatrix<double> mass_matrix_pressure;
    SparseMatrix<double> laplace_matrix_new_pressure;
    SparseMatrix<double> laplace_matrix_old_pressure;
    Vector<double>       system_rhs_pressure;
    Vector<double>       solution_pressure_new_iteration;
    Vector<double>       solution_pressure_old_iteration;
    Vector<double>       solution_pressure_initial_time;
    Vector<double>       old_solution_pressure;
    
    unsigned int timestep_number_max;
    unsigned int timestep_number;
    unsigned int refinement_level;
    double       time;
    double       time_step;
    double       time_max;
    double       theta_pressure;
    double       mass_balance_ratio;
    double       moisture_flow_inlet;
    double       moisture_flow_outlet;
    std::string  mesh_filename;
    bool use_mesh_file;
    
    Parameters::AllParameters<dim>  parameters;
  };

  template<int dim>
  Heat_Pipe<dim>::Heat_Pipe(int argc, char *argv[])
  :
  dof_handler_pressure(triangulation_pressure),
  fe_pressure(1)
  {
	  std::cout << "Program run with the following arguments:\n";
	  for (int i=0; i<argc; i++)
		  std::cout << "arg " << i << " : " << argv[i] << "\n";

	  std::string input_filename = argv[1];
	  std::cout << "parameter file: " << input_filename << "\n";

	  ParameterHandler prm;
	  Parameters::AllParameters<dim>::declare_parameters (prm);
	  prm.read_input (input_filename);
	  parameters.parse_parameters (prm);

	  theta_pressure      = parameters.theta;
	  timestep_number_max = parameters.timestep_number_max;
	  time_step           = parameters.time_step;
	  time_max            = time_step*timestep_number_max;
	  refinement_level    = parameters.refinement_level;
	  use_mesh_file       = parameters.use_mesh_file;
	  mesh_filename       = parameters.mesh_filename;

	  std::cout << "Solving problem with : \n"
			  << "\ttheta pressure     : " << theta_pressure << "\n"
			  << "\ttimestep_number_max: " << timestep_number_max << "\n"
			  << "\ttime_step          : " << time_step << "\n"
			  << "\ttime_max           : " << time_max  << "\n"
			  << "\trefinement_level   : " << refinement_level << "\n"
			  << "\tuse_mesh_file      : " << use_mesh_file << "\n"
			  << "\tmesh_filename      : " << mesh_filename << "\n";

	  timestep_number=0;
	  time=0;
	  mass_balance_ratio=0;
	  moisture_flow_inlet=0;
	  moisture_flow_outlet=0;
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  Heat_Pipe<dim>::~Heat_Pipe ()
  {
    dof_handler_pressure.clear ();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::calculate_mass_balance_ratio()
  {
	  QGauss<dim>quadrature_formula(1);
	  QGauss<dim-1>face_quadrature_formula(1);

	  FEValues<dim>fe_values(fe_pressure, quadrature_formula,
			  update_values|update_gradients|
			  update_JxW_values);
	  FEFaceValues<dim>fe_face_values(fe_pressure,face_quadrature_formula,
			  update_values|update_gradients|
			  update_quadrature_points|update_JxW_values);
	  const unsigned int dofs_per_cell  =fe_pressure.dofs_per_cell;

	  Vector<double> old_pressure_values;
	  Vector<double> new_pressure_values;
	  Vector<double> old_specific_moisture_capacity;
	  Vector<double> new_specific_moisture_capacity;
	  Vector<double> old_unsaturated_hydraulic_conductivity;
	  Vector<double> new_unsaturated_hydraulic_conductivity;
	  Vector<double> old_moisture_content;
	  Vector<double> new_moisture_content;

	  double face_boundary_indicator;

	  mass_balance_ratio=0;

	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler_pressure.begin_active(),
	  endc = dof_handler_pressure.end();
	  for (; cell!=endc; ++cell)
	  {
		  fe_values.reinit (cell);

		  old_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  new_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  old_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  new_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  old_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  new_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  old_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  new_moisture_content.reinit(cell->get_fe().dofs_per_cell);

		  cell->get_dof_values(solution_pressure_initial_time,old_pressure_values);
		  cell->get_dof_values(solution_pressure_old_iteration,new_pressure_values);

		  for (unsigned int i=0; i<dofs_per_cell; i++)
		  {
			  hydraulic_properties(
					  old_pressure_values[i],
					  old_specific_moisture_capacity[i],
					  old_unsaturated_hydraulic_conductivity[i],
					  old_moisture_content[i],
					  parameters.hydraulic_properties);
			  hydraulic_properties(
					  new_pressure_values[i],
					  new_specific_moisture_capacity[i],
					  new_unsaturated_hydraulic_conductivity[i],
					  new_moisture_content[i],
					  parameters.hydraulic_properties);
		  }

		  mass_balance_ratio+=
				  cell->diameter()*
				  (new_moisture_content[0]-old_moisture_content[0]);

		  for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
		  {
			  face_boundary_indicator = cell->face(face_number)->boundary_indicator();
			  if (cell->face(face_number)->at_boundary())
			  {
				  if (face_boundary_indicator==0)
				  {
					  mass_balance_ratio-=
							  0.5*cell->diameter()*
							  (new_moisture_content[0]-old_moisture_content[0]);

					  moisture_flow_inlet+=time_step*
							  0.5*(new_unsaturated_hydraulic_conductivity[0]+
									  new_unsaturated_hydraulic_conductivity[1])
									  *((1./cell->diameter())*(new_pressure_values[0]-new_pressure_values[1])+1);
				  }
				  else if (face_boundary_indicator==1)
				  {
					  mass_balance_ratio+=
							  0.5*cell->diameter()*
							  (new_moisture_content[1]-old_moisture_content[1]);

					  moisture_flow_outlet+=time_step*
							  0.5*(new_unsaturated_hydraulic_conductivity[0]+
									  new_unsaturated_hydraulic_conductivity[1])
									  *((1./cell->diameter())*(new_pressure_values[0]-new_pressure_values[1])+1);
				  }
			  }
		  }
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::read_grid_pressure()
  {
	  if(use_mesh_file)
	  {
		  GridIn<dim> grid_in;
		  grid_in.attach_triangulation (triangulation_pressure);
		  std::ifstream input_file (mesh_filename);
		  grid_in.read_msh (input_file);
		  dof_handler_pressure.distribute_dofs (fe_pressure);
	  }
	  else
	  {
		  GridGenerator::hyper_cube (triangulation_pressure,0,parameters.domain_size/*(cm)*/);
		  triangulation_pressure.refine_global (refinement_level);
		  dof_handler_pressure.distribute_dofs (fe_pressure);
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::setup_system_pressure()
  {
    hanging_node_constraints_pressure.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_pressure,
					    hanging_node_constraints_pressure);
    hanging_node_constraints_pressure.close();

    CompressedSparsityPattern csp(dof_handler_pressure.n_dofs(),
				  dof_handler_pressure.n_dofs());
    
    DoFTools::make_sparsity_pattern(dof_handler_pressure,csp);
        
    hanging_node_constraints_pressure.condense (csp);    
    sparsity_pattern_pressure.copy_from(csp);

    solution_pressure_new_iteration.reinit(dof_handler_pressure.n_dofs());
    solution_pressure_old_iteration.reinit(dof_handler_pressure.n_dofs());
    solution_pressure_initial_time.reinit(dof_handler_pressure.n_dofs());
    old_solution_pressure.reinit(dof_handler_pressure.n_dofs());
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_pressure()
  {  
	  system_rhs_pressure.reinit         (dof_handler_pressure.n_dofs());
	  system_matrix_pressure.reinit      (sparsity_pattern_pressure);
	  mass_matrix_pressure.reinit        (sparsity_pattern_pressure);
	  laplace_matrix_new_pressure.reinit (sparsity_pattern_pressure);
	  laplace_matrix_old_pressure.reinit (sparsity_pattern_pressure);

	  std::string quadrature_option;
	  unsigned int order=0;
	  if (parameters.lumped_matrix==false)
	  {
		  quadrature_option="gauss";
		  order=2;
	  }
  	  if (parameters.lumped_matrix==true)
		  quadrature_option="trapez";

  	  QuadratureSelector<dim> quadrature_formula(quadrature_option,order);

	  FEValues<dim>fe_values(fe_pressure, quadrature_formula,
			  update_values|update_gradients|
			  update_JxW_values);
	  const unsigned int dofs_per_cell  =fe_pressure.dofs_per_cell;
	  const unsigned int n_q_points     =quadrature_formula.size();

	  FullMatrix<double> cell_mass_matrix       (dofs_per_cell,dofs_per_cell);
	  FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell,dofs_per_cell);
	  FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell,dofs_per_cell);
	  Vector<double>     cell_rhs               (dofs_per_cell);

	  std::vector<unsigned int> local_dof_indices(fe_pressure.dofs_per_cell);
	  Vector<double> old_pressure_values;
	  Vector<double> new_pressure_values_old_iteration;

	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler_pressure.begin_active(),
	  endc = dof_handler_pressure.end();
	  for (; cell!=endc; ++cell)
	  {
		  fe_values.reinit (cell);
		  cell_mass_matrix       =0;
		  cell_laplace_matrix_new=0;
		  cell_laplace_matrix_old=0;
		  cell_rhs               =0;

		  old_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  new_pressure_values_old_iteration.reinit(cell->get_fe().dofs_per_cell);

		  cell->get_dof_values(old_solution_pressure,old_pressure_values);
		  cell->get_dof_values(solution_pressure_old_iteration,new_pressure_values_old_iteration);
		  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		  {
			  for (unsigned int k=0; k<dofs_per_cell; k++)
			  {
				  double old_specific_moisture_capacity=0;
				  double old_unsaturated_hydraulic_conductivity=0;
				  double old_moisture_content=0;
				  double new_specific_moisture_capacity=0;
				  double new_unsaturated_hydraulic_conductivity=0;
				  double new_moisture_content=0;

				  hydraulic_properties(
						  old_pressure_values[k],
						  old_specific_moisture_capacity,
						  old_unsaturated_hydraulic_conductivity,
						  old_moisture_content,
						  parameters.hydraulic_properties);
				  hydraulic_properties(
						  new_pressure_values_old_iteration[k],
						  new_specific_moisture_capacity,
						  new_unsaturated_hydraulic_conductivity,
						  new_moisture_content,
						  parameters.hydraulic_properties);

				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
					  for (unsigned int j=0; j<dofs_per_cell; ++j)
					  {
						  if (parameters.moisture_transport_equation.compare("head")==0)
						  {
							  cell_mass_matrix(i,j)+=
									  (theta_pressure)*
									  new_specific_moisture_capacity*
									  fe_values.shape_value(k,q_point)*
									  fe_values.shape_value(i,q_point)*
									  fe_values.shape_value(j,q_point)*
									  fe_values.JxW(q_point)
									  +
									  (1-theta_pressure)*
									  old_specific_moisture_capacity*
									  fe_values.shape_value(k,q_point)*
									  fe_values.shape_value(i,q_point)*
									  fe_values.shape_value(j,q_point)*
									  fe_values.JxW(q_point);
						  }
						  else if (parameters.moisture_transport_equation.compare("mixed")==0)
						  {
							  cell_mass_matrix(i,j)+=
									  new_specific_moisture_capacity*
									  fe_values.shape_value(k,q_point)*
									  fe_values.shape_value(j,q_point)*
									  fe_values.shape_value(i,q_point)*
									  fe_values.JxW(q_point);
						  }
						  else
						  {
							  std::cout << "Moisture transport equation \""
									  << parameters.moisture_transport_equation
									  << "\" is not implemented. Error.\n";
							  throw -1;
						  }
						  cell_laplace_matrix_new(i,j)+=(
								  new_unsaturated_hydraulic_conductivity*
								  fe_values.shape_value(k,q_point)*
								  fe_values.shape_grad(j,q_point)*
								  fe_values.shape_grad(i,q_point)*
								  fe_values.JxW(q_point));

						  cell_laplace_matrix_old(i,j)+=(
								  old_unsaturated_hydraulic_conductivity*
								  fe_values.shape_value(k,q_point)*
								  fe_values.shape_grad(j,q_point)*
								  fe_values.shape_grad(i,q_point)*
								  fe_values.JxW(q_point));
					  }
					  cell_rhs(i)+=
							  time_step*
							  (theta_pressure)*
							  new_unsaturated_hydraulic_conductivity*
							  fe_values.shape_value(k,q_point)*
							  fe_values.shape_grad(i,q_point)[0]*
							  fe_values.JxW(q_point)
							  +
							  time_step*
							  (1.-theta_pressure)*
							  old_unsaturated_hydraulic_conductivity*
							  fe_values.shape_value(k,q_point)*
							  fe_values.shape_grad(i,q_point)[0]*
							  fe_values.JxW(q_point);

					  if (parameters.moisture_transport_equation.compare("mixed")==0)
					  {
						  cell_rhs(i)-=(
								  new_moisture_content-
								  old_moisture_content)*
								  fe_values.shape_value(k,q_point)*
								  fe_values.shape_value(i,q_point)*
								  fe_values.JxW(q_point);
					  }
				  }
			  }
		  }
		  cell->get_dof_indices (local_dof_indices);
		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			  {
				  laplace_matrix_new_pressure
				  .add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_new(i,j));
				  laplace_matrix_old_pressure
				  .add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_old(i,j));
				  mass_matrix_pressure
				  .add(local_dof_indices[i],local_dof_indices[j],cell_mass_matrix(i,j));
			  }
			  system_rhs_pressure(local_dof_indices[i])+=cell_rhs(i);
		  }
	  }

	  double surface_pressure_bottom=parameters.bottom_fixed_value;
	  double surface_pressure_top   =parameters.top_fixed_value;
	  Vector<double> tmp(solution_pressure_new_iteration.size ());
	  if (parameters.moisture_transport_equation.compare("head")==0)
		  mass_matrix_pressure.vmult(tmp,old_solution_pressure);
	  else if (parameters.moisture_transport_equation.compare("mixed")==0)
		  mass_matrix_pressure.vmult(tmp,solution_pressure_old_iteration);
	  else
	  {
		  std::cout << "Moisture transport equation \""
				  << parameters.moisture_transport_equation
				  << "\" is not implemented. Error.\n";
		  throw -1;
	  }

	  system_rhs_pressure.add          ( 1.0,tmp);
	  laplace_matrix_old_pressure.vmult( tmp,old_solution_pressure);
	  system_rhs_pressure.add          (-(1-theta_pressure)*time_step,tmp);

	  system_matrix_pressure.copy_from(mass_matrix_pressure);
	  system_matrix_pressure.add      (theta_pressure*time_step, laplace_matrix_new_pressure);

	  hanging_node_constraints_pressure.condense(system_matrix_pressure);
	  hanging_node_constraints_pressure.condense(system_rhs_pressure);

	  std::map<unsigned int,double> boundary_values;
	  VectorTools::interpolate_boundary_values(dof_handler_pressure,
			  1,
			  ConstantFunction<dim>(surface_pressure_bottom),
			  boundary_values);
	  MatrixTools::apply_boundary_values (boundary_values,
			  system_matrix_pressure,
			  solution_pressure_new_iteration,
			  system_rhs_pressure);

	  boundary_values.clear();
	  VectorTools::interpolate_boundary_values(dof_handler_pressure,
			  0,
			  ConstantFunction<dim>(surface_pressure_top),
			  boundary_values);
	  MatrixTools::apply_boundary_values(boundary_values,
			  system_matrix_pressure,
			  solution_pressure_new_iteration,
			  system_rhs_pressure);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::solve_pressure ()
  {
    SolverControl solver_control(solution_pressure_new_iteration.size(),
				 1e-8*system_rhs_pressure.l2_norm ());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize (system_matrix_pressure, 1.2);

    cg.solve(system_matrix_pressure,solution_pressure_new_iteration,
	     system_rhs_pressure,preconditioner);
    
    hanging_node_constraints_pressure.distribute(solution_pressure_new_iteration);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::output_results () const
  {
    DataOut<dim> data_out;
    
    data_out.attach_dof_handler(dof_handler_pressure);
    data_out.add_data_vector(solution_pressure_new_iteration,"solution");
    data_out.build_patches();
    
    std::stringstream t;
    t << timestep_number;
    std::stringstream d;
    d << dim;
    std::stringstream ts;
    ts << time_step;

    std::string lm;
    if (parameters.lumped_matrix==true)
    lm="lumped_";
    
    std::string filename = "solution_"
    		+ parameters.moisture_transport_equation + "_" + lm
			+ d.str() + "d_"
			+ ts.str() + "_time_"
			+ t.str() + ".gp";
    
    std::ofstream output (filename.c_str());
    data_out.write_gnuplot (output);      
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
   void Heat_Pipe<dim>::initial_condition()
   {
	  VectorTools::project(dof_handler_pressure,
			  hanging_node_constraints_pressure,
			  QGauss<dim>(3),
			  ConstantFunction<dim> (parameters.initial_condition),
			  old_solution_pressure);
	  solution_pressure_new_iteration
	  =old_solution_pressure;
	  solution_pressure_old_iteration
	  =old_solution_pressure;
	  solution_pressure_initial_time
	  =old_solution_pressure;
   }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::run()
  {
	  read_grid_pressure();
	  setup_system_pressure();
	  initial_condition();
	  for (time=time_step, timestep_number=1;
			  time<=time_max;
			  time+=time_step, ++timestep_number)
	  {
		  std::cout << "Time step " << timestep_number << std::endl;

		  unsigned int iteration=0;
		  double tolerance_pressure=1000.;
		  do
		  {
			  iteration++;
			  assemble_system_pressure();
			  solve_pressure();

			  tolerance_pressure
			  =fabs(solution_pressure_new_iteration.l1_norm()-
					  solution_pressure_old_iteration.l1_norm())
					  /solution_pressure_new_iteration.l1_norm(); //(%)

			  solution_pressure_old_iteration
			  =solution_pressure_new_iteration;

			  if (iteration>1500 && iteration<1561)
				  std::cout << "iteration: " << iteration
				  << "\ttolerance: " << tolerance_pressure << "\n";
		  }
		  while (tolerance_pressure>0.0015);


		  calculate_mass_balance_ratio();
		  std::cout << "\tmoisture_balance: "
				  << mass_balance_ratio/(moisture_flow_inlet-moisture_flow_outlet)
				  << "\n";

		  if (timestep_number==timestep_number_max)
			  output_results();
		  old_solution_pressure
		  =solution_pressure_new_iteration;
	  }
	  std::cout << "\t Job Done!!"
			  << std::endl;
  }
}
//**************************************************************************************
//--------------------------------------------------------------------------------------
//**************************************************************************************
int main (int argc, char *argv[])
{
	try
	{
		using namespace TRL;
		using namespace dealii;
		{
			deallog.depth_console (0);
			Heat_Pipe<1> laplace_problem(argc,argv);
			laplace_problem.run();
		}
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}
	return 0;
}
