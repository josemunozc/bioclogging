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
#include <algorithm>
#include <time.h>

// Host dependent pathnames
#define stringify(Argument) #Argument
#define PrintHost(Host) {\
	std::cout << "Compiled on " stringify(Host) << std::endl;}

#ifndef HOST
	#define HOST unknown_host
#endif
#define HOST_OFFICE zerpiko
#define HOST_CLUSTER c1045890
#define HEADER_SURFACE_COEFFICIENTS(folder) </home/folder/libraries/SurfaceCoefficients.h>
#define HEADER_ANALYTICAL_SOLUTION(folder) </home/folder/libraries/AnalyticSolution.h>
#define HEADER_BOUNDARY_CONDITION(folder) </home/folder/libraries/BoundaryConditions.h>
#define HEADER_MATERIAL_DATA(folder) </home/folder/libraries/MaterialData.h>
#define HEADER_DATA_TOOLS(folder) </home/folder/libraries/data_tools.h>


#if HOST == HOST_CLUSTER
#define PATH HOST/code
#elif HOST == HOST_OFFICE
#define PATH HOST
#endif

#include HEADER_SURFACE_COEFFICIENTS(PATH)
#include HEADER_ANALYTICAL_SOLUTION(PATH)
#include HEADER_BOUNDARY_CONDITION(PATH)
#include HEADER_MATERIAL_DATA(PATH)
#include HEADER_DATA_TOOLS(PATH)

//class Hydraulic_Properties {
//public:
//
//	Hydraulic_Properties (
//			std::string type_of_hydraulic_properties_,
//			double moisture_content_saturation_,
//			double moisture_content_residual_,
//			double hydraulic_conductivity_saturated_);
//	double get_specific_moisture_capacity(double pressure_head_);
//
//private:
//	std::string type_of_hydraulic_properties;
//	double moisture_content_saturation;
//	double moisture_content_residual;
//	double hydraulic_conductivity_saturated;
//
//	double van_genuchten_alpha;
//	double van_genuchten_n;
//	double van_genuchten_m;
//};
//
//Hydraulic_Properties::Hydraulic_Properties(
//		std::string type_of_hydraulic_properties_,
//		double moisture_content_saturation_,
//		double moisture_content_residual_,
//		double hydraulic_conductivity_saturated_)
//{
//	type_of_hydraulic_properties=type_of_hydraulic_properties_;
//	moisture_content_saturation=moisture_content_saturation_;
//	moisture_content_residual=moisture_content_residual_;
//	hydraulic_conductivity_saturated=hydraulic_conductivity_saturated_;
//
//	van_genuchten_alpha;
//	van_genuchten_n;
//	van_genuchten_m;
//
//}

//double Hydraulic_Properties::get_specific_moisture_capacity(double pressure_head)
//{
//
//	if (type_of_hydraulic_properties.compare("haverkamp_et_al_1977")==0)
//	{
//		double alpha=1.611E6;
//		double beta =3.96;
//		double gamma=4.74;
//		double A=1.175E6;
//
//		return(-1.*alpha*(moisture_content_saturation-moisture_content_residual)
//				*beta*pressure_head*pow(fabs(pressure_head),beta-2)
//		/pow(alpha+pow(fabs(pressure_head),beta),2));
//
//	}
//	else if (type_of_hydraulic_properties.compare("van_genuchten_1980")==0)
//	{
//		double alpha
//		=parameters.van_genuchten_alpha;//0.0335;
//		double n
//		=parameters.van_genuchten_n;//2;
//		double m
//		=1.-1./n;
//
//		return(-1.*alpha*m*n*(moisture_content_saturation-moisture_content_residual)*
//				pow(alpha*fabs(pressure_head),n-1.)*pow(1.+pow(alpha*fabs(pressure_head),n),-1.*m-1.)*
//				pressure_head/fabs(pressure_head));
//	}
//	else
//	{
//		std::cout << "Equations for \"" << type_of_hydraulic_properties
//				<< "\" are not implemented. Error.\n";
//		throw -1;
//	}
//}



namespace TRL
{
  using namespace dealii;

#define HEADER_INITIAL_VALUE(folder) </home/folder/libraries/InitialValue.h>
#include HEADER_INITIAL_VALUE(PATH) 
#include "Parameters.h"

  template <int dim>
  class Heat_Pipe
  {
  public:
    Heat_Pipe(int argc, char *argv[]);
    ~Heat_Pipe();
    void run();

  private:
    void read_grid();
    void refine_grid(bool adaptive);
    void setup_system();
    void initial_condition();
    void assemble_system_flow();
    void assemble_system_transport();
    void solve();
    void output_results() const;

    void estimate_flow_velocities();

//    void biomass_concentration();

    void calculate_mass_balance_ratio();
    void hydraulic_properties(double pressure_head,
    		double &specific_moisture_capacity,
      		double &hydraulic_conductivity,
      		double &actual_total_moisture_content,
			double &actual_free_water_moisture_content,
      		std::string hydraulic_properties,
			double biomass_concentration=0);

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    FE_Q<dim>          fe;
    ConstraintMatrix   hanging_node_constraints;
    SparsityPattern    sparsity_pattern;

    // Richards' equation variables
    SparseMatrix<double> system_matrix_flow;
    SparseMatrix<double> mass_matrix_richards;
    SparseMatrix<double> laplace_matrix_new_richards;
    SparseMatrix<double> laplace_matrix_old_richards;
    Vector<double>       system_rhs_flow;
    Vector<double>       solution_flow_new_iteration;
    Vector<double>       solution_flow_old_iteration;
    Vector<double>       solution_flow_initial_time;
    Vector<double>       old_solution_flow;
    // Substrate variables
    SparseMatrix<double> system_matrix_transport;
    SparseMatrix<double> mass_matrix_transport_new;
    SparseMatrix<double> mass_matrix_transport_old;
    SparseMatrix<double> laplace_matrix_new_transport;
    SparseMatrix<double> laplace_matrix_old_transport;
    Vector<double>       system_rhs_transport;
    Vector<double>       solution_transport;
    Vector<double>       old_solution_transport;
    
    unsigned int timestep_number_max;
    unsigned int timestep_number;
    unsigned int refinement_level;
    double       time;
    double       time_step;
    double       time_max;
    double       theta_richards;
    double       theta_transport;
    double       change_in_moisture_from_beginning;
    double       change_in_biomass_from_beginning;
    double       moisture_flow_at_inlet_cumulative;
    double       moisture_flow_at_inlet_current;
    double       moisture_flow_at_outlet_cumulative;
    double       moisture_flow_at_outlet_current;
    double       total_moisture;
    double       total_biomass;
    std::string  mesh_filename;
    bool use_mesh_file;
    bool solve_flow;

    double milestone_time;
    double time_for_dry_conditions;
    double time_for_saturated_conditions;
    double time_for_homogeneous_concentration;
    bool steady_state_richards;
    bool transient_drying;
    bool transient_saturation;
    bool transient_transport;
    bool activate_transport;
    bool test_transport;
    bool test_biomass_apearance;
    double error_in_inlet_outlet_flow;
    double alternative_current_flow_at_inlet;
    double alternative_cumulative_flow_at_inlet;

    Vector<double> old_nodal_biomass_concentration;
    Vector<double> new_nodal_biomass_concentration;
    Vector<double> initial_nodal_biomass_concentration;
    Vector<double> old_nodal_total_moisture_content;
    Vector<double> new_nodal_total_moisture_content;
    Vector<double> old_nodal_free_moisture_content;
    Vector<double> new_nodal_free_moisture_content;
    Vector<double> new_nodal_hydraulic_conductivity;
    Vector<double> new_nodal_specific_moisture_capacity;
    Vector<double> new_nodal_flow_speed;
    Vector<double> old_nodal_flow_speed;
    Parameters::AllParameters<dim>  parameters;
  };

  template<int dim>
  Heat_Pipe<dim>::Heat_Pipe(int argc, char *argv[])
  :
  dof_handler(triangulation),
  fe(1)
  {
	  PrintHost(HOST);

	  std::cout << "Program run with the following arguments:\n";
	  for (int i=0; i<argc; i++)
		  std::cout << "arg " << i << " : " << argv[i] << "\n";

	  std::string input_filename = argv[1];
	  std::cout << "parameter file: " << input_filename << "\n";

	  ParameterHandler prm;
	  Parameters::AllParameters<dim>::declare_parameters (prm);
	  prm.read_input (input_filename);
	  parameters.parse_parameters (prm);

	  theta_richards      = parameters.theta_richards;
	  theta_transport     = parameters.theta_transport;
	  timestep_number_max = parameters.timestep_number_max;
	  time_step           = parameters.time_step;
	  time_max            = time_step*timestep_number_max;
	  refinement_level    = parameters.refinement_level;
	  use_mesh_file       = parameters.use_mesh_file;
	  mesh_filename       = parameters.mesh_filename;

	  timestep_number=0;
	  time=0;
	  change_in_moisture_from_beginning=0;
	  change_in_biomass_from_beginning=0;
	  moisture_flow_at_inlet_current=0;
	  moisture_flow_at_inlet_cumulative=0;
	  moisture_flow_at_outlet_current=0;
	  moisture_flow_at_outlet_cumulative=0;
	  total_moisture=0;
	  total_biomass=0;
	  solve_flow=true;

	  milestone_time=0;
	  time_for_dry_conditions=0;
	  time_for_saturated_conditions=0;
	  time_for_homogeneous_concentration=0;
	  steady_state_richards=false;
	  transient_drying=true;
	  transient_saturation=false;
	  transient_transport=false;
	  activate_transport=false;
	  test_transport=parameters.test_function_transport;
	  test_biomass_apearance=true;
	  error_in_inlet_outlet_flow=0.;
	  alternative_current_flow_at_inlet=0.;
	  alternative_cumulative_flow_at_inlet=0.;
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  Heat_Pipe<dim>::~Heat_Pipe ()
  {
    dof_handler.clear ();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::hydraulic_properties(
		  double pressure_head,
		  double &specific_moisture_capacity,
		  double &hydraulic_conductivity,
		  double &actual_total_moisture_content,
		  double &actual_free_water_moisture_content,
		  std::string hydraulic_properties,
		  double biomass_concentration)

		  {
	  double moisture_content_saturation
	  =parameters.moisture_content_saturation;//0.368;
	  double moisture_content_residual
	  =parameters.moisture_content_residual;//0.102;
	  double hydraulic_conductivity_saturated
	  =parameters.saturated_hydraulic_conductivity;//0.00922; // (cm/s)

	  if (hydraulic_properties.compare("haverkamp_et_al_1977")==0)
	  {
		  double alpha=1.611E6;
		  double beta =3.96;
		  double gamma=4.74;
		  double A=1.175E6;

		  specific_moisture_capacity
		  =-1.*alpha*(moisture_content_saturation-moisture_content_residual)
		  *beta*pressure_head*pow(fabs(pressure_head),beta-2)
		  /pow(alpha+pow(fabs(pressure_head),beta),2);

		  hydraulic_conductivity
		  =hydraulic_conductivity_saturated*A/(A+pow(fabs(pressure_head),gamma));

		  actual_total_moisture_content
		  =(alpha*(moisture_content_saturation-moisture_content_residual)/
				  (alpha+pow(fabs(pressure_head),beta)))+moisture_content_residual;
	  }
	  else if (hydraulic_properties.compare("van_genuchten_1980")==0)
	  {
		  double alpha
		  =parameters.van_genuchten_alpha;//0.0335;
		  double n
		  =parameters.van_genuchten_n;//2;
		  double m
		  =1.-1./n;
		  double saturation_residual
		    =moisture_content_residual/parameters.porosity;
		  /* * * * * * *
		   * total water
		   * * * * * * */
		  double effective_total_saturation
		  =1./pow(1.+pow(alpha*fabs(pressure_head),n),m);
		  if (pressure_head>=0.)
		    effective_total_saturation=1.;

		  actual_total_moisture_content
		  =(moisture_content_saturation-moisture_content_residual)*effective_total_saturation
				  +moisture_content_residual;
		  double actual_total_saturation // cm3_total_water/cm3_void
		  =actual_total_moisture_content/parameters.porosity;

		  /* * * * * * * *
		   * bacteria
		   * * * * * * * */
		  double actual_biomass_saturation // cm3_biomass/cm3_void
		  =actual_total_saturation*(biomass_concentration/parameters.biomass_dry_density);
		  
		  double effective_biomass_saturation
		  =actual_biomass_saturation/(1.-saturation_residual);
		  if (effective_biomass_saturation>=1.)
		    effective_biomass_saturation=1.;

//		  double actual_free_water_saturation=0.; // cm3_free_water/cm3_void
		  if (effective_total_saturation<=effective_biomass_saturation)
		    {
		      effective_total_saturation=effective_biomass_saturation;
//		      actual_free_water_saturation=saturation_residual;
		    }
//		  else
//		    actual_free_water_saturation // cm3_free_water/cm3_void
//		      =actual_total_saturation-actual_biomass_saturation;

//		  if (actual_free_water_saturation<=saturation_residual)
//		    actual_free_water_saturation=saturation_residual;
//		  actual_free_water_moisture_content
//		  =actual_free_water_saturation*parameters.porosity;
		    
		  /* * * * * * * *
		   * free water
		   * * * * * * * */
		   double actual_free_water_saturation // cm3_free_water/cm3_void
		   =actual_total_saturation-actual_biomass_saturation;

		   if (actual_free_water_saturation<=0.)
			   actual_free_water_saturation=0.;

//		  if (actual_free_water_saturation<=saturation_residual)
//			  actual_free_water_saturation=saturation_residual;
		   actual_free_water_moisture_content
		   =actual_free_water_saturation*parameters.porosity;

		  /* * * * * * * * * * * *
		   * hydraulic properties
		   * * * * * * * * * * * */
		  specific_moisture_capacity
		  =-1.*alpha*m*n*(moisture_content_saturation-moisture_content_residual)*
		  pow(alpha*fabs(pressure_head),n-1.)*pow(1.+pow(alpha*fabs(pressure_head),n),-1.*m-1.)*
		  pressure_head/fabs(pressure_head);

		  double relative_permeability_mualem
		  =pow(effective_total_saturation,0.5)*
		  pow(pow(1.-pow(effective_biomass_saturation,1./m),m)-
				  pow(1.-pow(effective_total_saturation,1./m),m),2.);

		  hydraulic_conductivity
		  =hydraulic_conductivity_saturated*
		  relative_permeability_mualem;

		  /*
		   * Check values out of range
		   * */
//		  if (specific_moisture_capacity<0. ||
//				  relative_permeability_mualem<0. ||
//				  relative_permeability_mualem>1. ||
//				  actual_total_moisture_content<0. ||
//				  actual_free_water_moisture_content<0. ||
//				  actual_free_water_moisture_content>1. ||
//				  pressure_head>0.||
//				  biomass_concentration<0. ||
//				  effective_biomass_saturation>effective_total_saturation)
//		  {
//			  std::cout << "\tspecific_moisture_capacity: "       << specific_moisture_capacity << "\n"
//					  << "\trelative_permeability_mualem: "       << relative_permeability_mualem << "\n"
//					  << "\tactual_total_moisture_content: "      << actual_total_moisture_content << "\n"
//					  << "\tactual_free_water_moisture_content: " << actual_free_water_moisture_content << "\n"
//					  << "\tpressure_head: "                      << pressure_head << "\n"
//					  << "\tbiomass_concentration: "              << biomass_concentration << "\n"
//					  << "\teffective_biomass_saturation: "       << effective_biomass_saturation << "\n"
//					  << "\teffective_total_saturation: "         << effective_total_saturation << "\n"
//					  << "\tsaturation_residual: "                << saturation_residual << "\n";
//			  throw -1;
//		  }
	  }
	  else
	  {
		  std::cout << "Equations for \"" << hydraulic_properties
				  << "\" are not implemented. Error.\n";
		  throw -1;
	  }
		  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::estimate_flow_velocities()
  {
	  QGauss<dim>quadrature_formula(1);
	  QGauss<dim-1>face_quadrature_formula(1);

	  FEValues<dim>fe_values(fe, quadrature_formula,
			  update_values|update_gradients|
			  update_JxW_values);
	  FEFaceValues<dim>fe_face_values(fe,face_quadrature_formula,
			  update_values|update_gradients|
			  update_quadrature_points|update_JxW_values);
	  const unsigned int dofs_per_cell  =fe.dofs_per_cell;

	  Vector<double> new_pressure_values;
	  Vector<double> new_specific_moisture_capacity;
	  Vector<double> new_unsaturated_hydraulic_conductivity;
	  Vector<double> new_total_moisture_content;
	  Vector<double> new_free_moisture_content;

	  Vector<double> old_pressure_values;
	  Vector<double> old_specific_moisture_capacity;
	  Vector<double> old_unsaturated_hydraulic_conductivity;
	  Vector<double> old_total_moisture_content;
	  Vector<double> old_free_moisture_content;

	  double new_average_pressure_previous_cell=0.;
	  double old_average_pressure_previous_cell=0.;
	  unsigned int cell_integer_index=0;
	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell)
	  {
		  fe_values.reinit (cell);

		  new_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  new_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  new_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  new_total_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  new_free_moisture_content.reinit(cell->get_fe().dofs_per_cell);

		  old_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  old_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  old_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  old_total_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  old_free_moisture_content.reinit(cell->get_fe().dofs_per_cell);

		  cell->get_dof_values(solution_flow_old_iteration,new_pressure_values);
		  cell->get_dof_values(old_solution_flow,old_pressure_values);

		  for (unsigned int i=0; i<dofs_per_cell; i++)
		  {
			  hydraulic_properties(
					  new_pressure_values[i],
					  new_specific_moisture_capacity[i],
					  new_unsaturated_hydraulic_conductivity[i],
					  new_total_moisture_content[i],
					  new_free_moisture_content[i],
					  parameters.hydraulic_properties,
					  new_nodal_biomass_concentration[cell_integer_index+i]);
			  hydraulic_properties(
					  old_pressure_values[i],
					  old_specific_moisture_capacity[i],
					  old_unsaturated_hydraulic_conductivity[i],
					  old_total_moisture_content[i],
					  old_free_moisture_content[i],
					  parameters.hydraulic_properties,
					  old_nodal_biomass_concentration[cell_integer_index+i]);
		  }


		  if (cell_integer_index==0) // x=-100m fixed pressure
		  {
			  double old_extrapolated_pressure
			  =old_pressure_values[1]+1.5*(old_pressure_values[0]-old_pressure_values[1]);


			  old_nodal_flow_speed[cell_integer_index]=
			    -1.*old_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(old_pressure_values[1]+old_pressure_values[0])
			      -old_extrapolated_pressure)/cell->diameter()+1.);
			  
			  new_nodal_flow_speed[cell_integer_index]=
			    -1.*new_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(new_pressure_values[1]+new_pressure_values[0])
			      -new_pressure_values[0])/(0.5*cell->diameter())+1.);
		  }
		  else if (cell_integer_index==triangulation.n_active_cells()-1)
		  {
			  old_nodal_flow_speed[cell_integer_index]=
			    -1.*old_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(old_pressure_values[1]+old_pressure_values[0])
			      -old_average_pressure_previous_cell)/(cell->diameter())+1.);
			  
			  new_nodal_flow_speed[cell_integer_index]=
			    -1.*new_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(new_pressure_values[1]+new_pressure_values[0])
			      -new_average_pressure_previous_cell)/cell->diameter()+1.);
			  
			  double new_extrapolated_pressure
			  =new_pressure_values[0]+1.5*(new_pressure_values[1]-new_pressure_values[0]);
			  double old_extrapolated_pressure
			  =old_pressure_values[0]+1.5*(old_pressure_values[1]-old_pressure_values[0]);

			  old_nodal_flow_speed[cell_integer_index+1]
			    =-1.*old_unsaturated_hydraulic_conductivity[1]*
				((old_extrapolated_pressure-old_average_pressure_previous_cell)/cell->diameter()+1.);
			  
			  new_nodal_flow_speed[cell_integer_index+1]
			    =-1.*new_unsaturated_hydraulic_conductivity[1]*
			    ((new_extrapolated_pressure-new_average_pressure_previous_cell)/cell->diameter()+1.);
			  
			  // old_nodal_flow_speed[cell_integer_index+1]=
			  // 		  -1.*parameters.richards_top_flow_value;
			  // new_nodal_flow_speed[cell_integer_index+1]=
			  // 		  -1.*parameters.richards_top_flow_value;
		  }
		  else
		  {
			  old_nodal_flow_speed[cell_integer_index]=
			    -1.*old_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(old_pressure_values[1]+old_pressure_values[0])
			      -old_average_pressure_previous_cell)/(cell->diameter())+1.);
			  
			  new_nodal_flow_speed[cell_integer_index]=
			    -1.*new_unsaturated_hydraulic_conductivity[0]*
			    ((0.5*(new_pressure_values[1]+new_pressure_values[0])
			      -new_average_pressure_previous_cell)/cell->diameter()+1.);
		  }

		  new_average_pressure_previous_cell
		  =0.5*(new_pressure_values[1]+new_pressure_values[0]);
		  old_average_pressure_previous_cell
		  =0.5*(old_pressure_values[1]+old_pressure_values[0]);

		  cell_integer_index++;
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::calculate_mass_balance_ratio()
  {
	  QGauss<dim>quadrature_formula(1);
	  QGauss<dim-1>face_quadrature_formula(1);

	  FEValues<dim>fe_values(fe, quadrature_formula,
			  update_values|update_gradients|
			  update_JxW_values);
	  FEFaceValues<dim>fe_face_values(fe,face_quadrature_formula,
			  update_values|update_gradients|
			  update_quadrature_points|update_JxW_values);
	  const unsigned int dofs_per_cell  =fe.dofs_per_cell;

	  //Vector<double> new_transport_values;
	  Vector<double> old_transport_values;

	  Vector<double> initial_flow_values;
	  Vector<double> new_flow_values_old_iteration;
	  Vector<double> old_flow_values;

	  Vector<double> initial_specific_moisture_capacity;
	  Vector<double> new_specific_moisture_capacity;
	  Vector<double> old_specific_moisture_capacity;

	  Vector<double> initial_unsaturated_hydraulic_conductivity;
	  Vector<double> new_unsaturated_hydraulic_conductivity;
	  Vector<double> old_unsaturated_hydraulic_conductivity;

	  Vector<double> initial_total_moisture_content;
	  Vector<double> new_total_moisture_content;
	  Vector<double> old_total_moisture_content;

	  Vector<double> initial_free_moisture_content;
	  Vector<double> new_free_moisture_content;
	  Vector<double> old_free_moisture_content;

	  Vector<double> old_biomass_concentration_in_cell;
	  Vector<double> new_biomass_concentration_in_cell;
	  Vector<double> initial_biomass_concentration_in_cell;

	  double face_boundary_indicator;

	  change_in_moisture_from_beginning=0;
	  change_in_biomass_from_beginning=0;
	  total_moisture=0;
	  total_biomass=0;

	  unsigned int cell_integer_index=0;
	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell)
	  {
		  fe_values.reinit (cell);

		  //new_transport_values.reinit(cell->get_fe().dofs_per_cell);
		  old_transport_values.reinit(cell->get_fe().dofs_per_cell);

		  initial_flow_values.reinit(cell->get_fe().dofs_per_cell);
		  new_flow_values_old_iteration.reinit(cell->get_fe().dofs_per_cell);
		  old_flow_values.reinit(cell->get_fe().dofs_per_cell);

		  initial_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  new_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
		  old_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);

		  initial_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  new_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
		  old_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);

		  initial_total_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  new_total_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  old_total_moisture_content.reinit(cell->get_fe().dofs_per_cell);

		  initial_free_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  new_free_moisture_content.reinit(cell->get_fe().dofs_per_cell);
		  old_free_moisture_content.reinit(cell->get_fe().dofs_per_cell);

		  old_biomass_concentration_in_cell.reinit(cell->get_fe().dofs_per_cell);
		  new_biomass_concentration_in_cell.reinit(cell->get_fe().dofs_per_cell);
		  initial_biomass_concentration_in_cell.reinit(cell->get_fe().dofs_per_cell);

		  //cell->get_dof_values(solution_transport,new_transport_values);
		  cell->get_dof_values(old_solution_transport,old_transport_values);

		  cell->get_dof_values(solution_flow_initial_time,initial_flow_values);
		  cell->get_dof_values(solution_flow_old_iteration,new_flow_values_old_iteration);
		  cell->get_dof_values(old_solution_flow,old_flow_values);

		  for (unsigned int i=0; i<dofs_per_cell; i++)
		  {
		    if (transient_drying==false)
		      {
			//				  double new_substrate=0;
			//				  if (new_transport_values[i]>0)
			//					  new_substrate=new_transport_values[i];
			double old_substrate=0;
			if (old_transport_values[i]>0)
			  old_substrate=old_transport_values[i];
			
			// double old_substrate=0;
			// if (old_transport_values[0]>0)
			//   old_substrate+=0.5*old_transport_values[0];
			// if (old_transport_values[1]>0)
			//   old_substrate+=0.5*old_transport_values[1];

			
			/* *
				   * Some of the variables for the transport equation defined in the input file
				   * are provided in [mg_substrate/L_free_water]. They need to be transformed
				   * to [mg_substrate/cm3_soil] to be consistent with the primary variable in
				   * the transport equation. This is done in this way:
				   *
				   * half_velocity_constant[mg_substrate/L_free_water]
				   * =half_velocity_constant[mg_substrate/L_free_water]*
				   * moisture_content_free_water[cm3_free_water/cm3_soil]*
				   * free_water_volume_ratio [L_free_water/1000 cm3_free_water]
				   * =(1./1000.)*half_velocity_constant[mg_substrate/cm3_soil]
				   * */
				  hydraulic_properties(
						  old_flow_values[i],
						  old_specific_moisture_capacity[i],
						  old_unsaturated_hydraulic_conductivity[i],
						  old_total_moisture_content[i],
						  old_free_moisture_content[i],
						  parameters.hydraulic_properties,
						  old_nodal_biomass_concentration[cell_integer_index+i]);

				  new_nodal_biomass_concentration[cell_integer_index+i]=
						  old_nodal_biomass_concentration[cell_integer_index+i]*
						  exp((parameters.yield_coefficient*parameters.maximum_substrate_use_rate*
								  old_substrate/(old_substrate+(1./1000.)*parameters.half_velocity_constant)
								  -parameters.decay_rate)*time_step);
			  }
			  new_biomass_concentration_in_cell[i]=
					  new_nodal_biomass_concentration[cell_integer_index+i];
			  initial_biomass_concentration_in_cell[i]=
					  initial_nodal_biomass_concentration[cell_integer_index+i];

			  hydraulic_properties(
					  initial_flow_values[i],
					  initial_specific_moisture_capacity[i],
					  initial_unsaturated_hydraulic_conductivity[i],
					  initial_total_moisture_content[i],
					  initial_free_moisture_content[i],
					  parameters.hydraulic_properties,
					  initial_biomass_concentration_in_cell[i]);
			  hydraulic_properties(
					  new_flow_values_old_iteration[i],
					  new_specific_moisture_capacity[i],
					  new_unsaturated_hydraulic_conductivity[i],
					  new_total_moisture_content[i],
					  new_free_moisture_content[i],
					  parameters.hydraulic_properties,
					  new_biomass_concentration_in_cell[i]);

			  new_nodal_total_moisture_content[cell_integer_index+i]=
					  new_total_moisture_content[i];
			  new_nodal_free_moisture_content[cell_integer_index+i]=
					  new_free_moisture_content[i];
			  new_nodal_hydraulic_conductivity[cell_integer_index+i]=
					  new_unsaturated_hydraulic_conductivity[i];
			  new_nodal_specific_moisture_capacity[cell_integer_index+i]=
					  new_specific_moisture_capacity[i];
		  }
		  cell_integer_index++;

		  change_in_moisture_from_beginning+=
				  cell->diameter()*
				  (new_total_moisture_content[0]-initial_total_moisture_content[0]);

		  change_in_biomass_from_beginning+=
				  cell->diameter()*
				  (new_total_moisture_content[0]*new_biomass_concentration_in_cell[0]-
						  initial_total_moisture_content[0]*initial_biomass_concentration_in_cell[0])/
						  parameters.biomass_dry_density;

		  total_moisture+=
				  cell->diameter()*
				  new_total_moisture_content[0];

		  total_biomass+=
				  cell->diameter()*
				  new_total_moisture_content[0]*
				  new_biomass_concentration_in_cell[0]/
				  parameters.biomass_dry_density;

		  for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
		  {
			  face_boundary_indicator = cell->face(face_number)->boundary_indicator();
			  if (cell->face(face_number)->at_boundary())
			  {
				  if (face_boundary_indicator==0)
				  {
					  change_in_moisture_from_beginning-=
							  0.5*cell->diameter()*
							  (new_total_moisture_content[0]-initial_total_moisture_content[0]);

					  change_in_biomass_from_beginning-=
							  0.5*cell->diameter()*
							  (new_total_moisture_content[0]*new_biomass_concentration_in_cell[0]-
									  initial_total_moisture_content[0]*initial_biomass_concentration_in_cell[0])/
									  parameters.biomass_dry_density;

					  total_moisture-=
							  0.5*cell->diameter()*new_total_moisture_content[0];

					  total_biomass-=
							  0.5*cell->diameter()*
							  new_total_moisture_content[0]*
							  new_biomass_concentration_in_cell[0]/
							  parameters.biomass_dry_density;

					  moisture_flow_at_outlet_current=time_step*
							  0.5*(new_unsaturated_hydraulic_conductivity[0]+
									  new_unsaturated_hydraulic_conductivity[1])*
									  ((1./cell->diameter())*
											  (new_flow_values_old_iteration[1]-
													  new_flow_values_old_iteration[0])+1);
				  }
				  else if (face_boundary_indicator==1)
				  {
					  change_in_moisture_from_beginning+=
							  0.5*cell->diameter()*
							  (new_total_moisture_content[1]-initial_total_moisture_content[1]);

					  change_in_biomass_from_beginning+=
							  0.5*cell->diameter()*
							  (new_total_moisture_content[1]*new_biomass_concentration_in_cell[1]-
									  initial_total_moisture_content[1]*initial_biomass_concentration_in_cell[1])/
									  parameters.biomass_dry_density;

					  total_moisture+=
							  0.5*cell->diameter()*new_total_moisture_content[1];

					  total_biomass+=
							  0.5*cell->diameter()*
							  new_total_moisture_content[1]*
							  new_biomass_concentration_in_cell[1]/
							  parameters.biomass_dry_density;

					  if (transient_drying==false)
					  {
						  moisture_flow_at_inlet_current=time_step*
								  parameters.richards_top_flow_value;

						  alternative_current_flow_at_inlet=time_step*
								  0.5*(new_unsaturated_hydraulic_conductivity[0]+
										  new_unsaturated_hydraulic_conductivity[1])*
										  ((1./cell->diameter())*
												  (new_flow_values_old_iteration[1]-
														  new_flow_values_old_iteration[0])+1);
					  }

				  }
			  }
		  }
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::read_grid()
  {
	  if(use_mesh_file)
	  {
		  GridIn<dim> grid_in;
		  grid_in.attach_triangulation(triangulation);
		  std::ifstream input_file(mesh_filename);
		  grid_in.read_msh(input_file);
	  }
	  else
	  {
		  GridGenerator::hyper_cube(triangulation,-1.*parameters.domain_size/*(cm)*/,0);
		  triangulation.refine_global(refinement_level);
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::refine_grid(bool adaptive)
  {
//	   Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
//	   KellyErrorEstimator<dim>::estimate(dof_handler,
//			   QGauss<dim-1>(3),
//			   typename FunctionMap<dim>::type(),
//			   old_solution_richards,// solution_richards_new_iteration,
//			   estimated_error_per_cell);

	  /*
	   * We need to track the advance of the flow front. For this
	   * we estimate the 'gradient' in the pressure head
	   * */
	  std::vector<double> gradient;
	  {
		  QGauss<dim> quadrature_formula(2);
		  FEValues<dim> fe_values(fe,quadrature_formula,
				  update_values | update_gradients |
				  update_JxW_values);
		  const unsigned int n_q_points     =quadrature_formula.size();
		  std::vector<double> pressure_values(n_q_points);
		  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  endc = dof_handler.end();
		  for (; cell!=endc; ++cell)
		  {
			  fe_values.reinit (cell);
			  fe_values.get_function_values(solution_flow_new_iteration,
					  pressure_values);
			  gradient
			  .push_back(fabs(pressure_values[1]-pressure_values[0])
					  /cell->diameter());
		  }
	  }
	  std::sort(gradient.begin(),gradient.end());
	  double refinement_threshold
	  =gradient[(int)(0.90*gradient.size())];

	  {
		  QGauss<dim> quadrature_formula(2);
		  FEValues<dim> fe_values(fe,quadrature_formula,
				  update_values | update_gradients |
				  update_JxW_values);
		  const unsigned int n_q_points     =quadrature_formula.size();
		  std::vector<double> pressure_values(n_q_points);
		  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  endc = dof_handler.end();
		  for (; cell!=endc; ++cell)
		  {
			  fe_values.reinit (cell);
			  fe_values.get_function_values(solution_flow_new_iteration,
					  pressure_values);
			  if (adaptive==false)
				  cell->set_refine_flag();
			  else if (adaptive==true)
				  if (((fabs(pressure_values[1]-pressure_values[0])/cell->diameter())
						 >refinement_threshold)
						  /*&&
						  ((pressure_old_values[1]-pressure_old_values[0])>0)*/)
					  cell->set_refine_flag();

		  }
	  }

//	  if (adaptive)
//		  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
//				  estimated_error_per_cell,
//				  0.1, 0.1);
//	  else
//	  {
//		  //unsigned int cell_to_be_refined=0;
//		  typename DoFHandler<dim>::active_cell_iterator
//		  cell = dof_handler.begin_active(),
//		  endc = dof_handler.end();
//		  for (; cell!=endc; ++cell)
//		  {
//			  //		   if (cell->center()[0]>-10)
//			  cell->set_refine_flag();
//		  }
//	  }

	   std::vector<Vector<double> > transfer_in;
	   transfer_in.push_back(old_solution_flow);
	   transfer_in.push_back(solution_flow_new_iteration);
	   transfer_in.push_back(solution_flow_old_iteration);
	   transfer_in.push_back(solution_flow_initial_time);
	   transfer_in.push_back(old_solution_transport);
	   transfer_in.push_back(solution_transport);
	   transfer_in.push_back(old_nodal_biomass_concentration);
	   transfer_in.push_back(new_nodal_biomass_concentration);

	   SolutionTransfer<dim> solution_transfer(dof_handler);

	   triangulation.prepare_coarsening_and_refinement();
	   solution_transfer.prepare_for_coarsening_and_refinement(transfer_in);
	   triangulation.execute_coarsening_and_refinement();

	   setup_system();

	   std::vector<Vector<double> > transfer_out(8);
	   transfer_out[0].reinit(dof_handler.n_dofs());
	   transfer_out[1].reinit(dof_handler.n_dofs());
	   transfer_out[2].reinit(dof_handler.n_dofs());
	   transfer_out[3].reinit(dof_handler.n_dofs());
	   transfer_out[4].reinit(dof_handler.n_dofs());
	   transfer_out[5].reinit(dof_handler.n_dofs());
	   transfer_out[6].reinit(dof_handler.n_dofs());
	   transfer_out[7].reinit(dof_handler.n_dofs());

	   solution_transfer.interpolate(transfer_in,transfer_out);
	   old_solution_flow          =transfer_out[0];
	   solution_flow_new_iteration=transfer_out[1];
	   solution_flow_old_iteration=transfer_out[2];
	   solution_flow_initial_time =transfer_out[3];
	   old_solution_transport     =transfer_out[4];
	   solution_transport         =transfer_out[5];
	   old_nodal_biomass_concentration=transfer_out[6];
	   new_nodal_biomass_concentration=transfer_out[7];
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::setup_system()
  {
	  dof_handler.distribute_dofs(fe);
	  hanging_node_constraints.clear();
	  DoFTools::make_hanging_node_constraints(dof_handler,
			  hanging_node_constraints);
	  hanging_node_constraints.close();

	  CompressedSparsityPattern csp(dof_handler.n_dofs(),
			  dof_handler.n_dofs());

	  DoFTools::make_sparsity_pattern(dof_handler,csp);

	  hanging_node_constraints.condense(csp);
	  sparsity_pattern.copy_from(csp);

	  solution_flow_new_iteration.reinit(dof_handler.n_dofs());
	  solution_flow_old_iteration.reinit(dof_handler.n_dofs());
	  solution_flow_initial_time.reinit(dof_handler.n_dofs());
	  old_solution_flow.reinit(dof_handler.n_dofs());

	  solution_transport.reinit(dof_handler.n_dofs());
	  old_solution_transport.reinit(dof_handler.n_dofs());

	  old_nodal_biomass_concentration.reinit(dof_handler.n_dofs());
	  new_nodal_biomass_concentration.reinit(dof_handler.n_dofs());
	  initial_nodal_biomass_concentration.reinit(dof_handler.n_dofs());

	  new_nodal_total_moisture_content.reinit(dof_handler.n_dofs());
	  new_nodal_free_moisture_content.reinit(dof_handler.n_dofs());
	  new_nodal_hydraulic_conductivity.reinit(dof_handler.n_dofs());
	  new_nodal_specific_moisture_capacity.reinit(dof_handler.n_dofs());

	  new_nodal_flow_speed.reinit(dof_handler.n_dofs());
	  old_nodal_flow_speed.reinit(dof_handler.n_dofs());
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_transport()
  {
    system_rhs_transport.reinit        (dof_handler.n_dofs());
    system_matrix_transport.reinit     (sparsity_pattern);
    mass_matrix_transport_new.reinit   (sparsity_pattern);
    mass_matrix_transport_old.reinit   (sparsity_pattern);
    laplace_matrix_new_transport.reinit(sparsity_pattern);
    laplace_matrix_old_transport.reinit(sparsity_pattern);

    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);
    FEValues<dim> fe_values(fe,quadrature_formula,
    		update_values | update_gradients |
			update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
    		update_values|update_gradients|
			update_normal_vectors|
			update_quadrature_points|update_JxW_values);

    const unsigned int dofs_per_cell  =fe.dofs_per_cell;
    const unsigned int n_q_points     =quadrature_formula.size();
    const unsigned int n_face_q_points=face_quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix_new   (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_mass_matrix_old   (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell,dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);
    Vector<double> substrate_old_values;

    Vector<double> new_pressure_values;
    Vector<double> new_specific_moisture_capacity;
    Vector<double> new_unsaturated_hydraulic_conductivity;
    Vector<double> new_moisture_content_total_water;
    Vector<double> new_moisture_content_free_water;
    Vector<double> new_biomass_concentration_in_cell;

    Vector<double> old_pressure_values;
    Vector<double> old_specific_moisture_capacity;
    Vector<double> old_unsaturated_hydraulic_conductivity;
    Vector<double> old_moisture_content_total_water;
    Vector<double> old_moisture_content_free_water;
    Vector<double> old_biomass_concentration_in_cell;

    unsigned int cell_integer_index=0;
    double face_boundary_indicator;
	Tensor<1,dim> advection_field;
	advection_field[0]=-1.;

    typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
    	fe_values.reinit (cell);
    	cell_mass_matrix_new=0;
    	cell_mass_matrix_old=0;
    	cell_laplace_matrix_new=0;
    	cell_laplace_matrix_old=0;
    	cell_rhs=0;

    	substrate_old_values.reinit(cell->get_fe().dofs_per_cell);

    	new_pressure_values.reinit(cell->get_fe().dofs_per_cell);
    	new_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
    	new_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
    	new_moisture_content_total_water.reinit(cell->get_fe().dofs_per_cell);
    	new_moisture_content_free_water.reinit(cell->get_fe().dofs_per_cell);
    	new_biomass_concentration_in_cell.reinit(cell->get_fe().dofs_per_cell);

    	old_pressure_values.reinit(cell->get_fe().dofs_per_cell);
    	old_specific_moisture_capacity.reinit(cell->get_fe().dofs_per_cell);
    	old_unsaturated_hydraulic_conductivity.reinit(cell->get_fe().dofs_per_cell);
    	old_moisture_content_total_water.reinit(cell->get_fe().dofs_per_cell);
    	old_moisture_content_free_water.reinit(cell->get_fe().dofs_per_cell);
    	old_biomass_concentration_in_cell.reinit(cell->get_fe().dofs_per_cell);

    	cell->get_dof_values(old_solution_transport,substrate_old_values);
    	cell->get_dof_values(old_solution_flow,old_pressure_values);
    	cell->get_dof_values(solution_flow_old_iteration,new_pressure_values);

		for (unsigned int k=0; k<dofs_per_cell; k++)
		{
			if (transient_drying==false)
			{
				old_biomass_concentration_in_cell
				=old_nodal_biomass_concentration[cell_integer_index+k];
				new_biomass_concentration_in_cell
				=new_nodal_biomass_concentration[cell_integer_index+k];
			}

			hydraulic_properties(
					old_pressure_values[k],
					old_specific_moisture_capacity[k],
					old_unsaturated_hydraulic_conductivity[k],
					old_moisture_content_total_water[k],
					old_moisture_content_free_water[k],
					parameters.hydraulic_properties,
					old_biomass_concentration_in_cell[k]);
			hydraulic_properties(
					new_pressure_values[k],
					new_specific_moisture_capacity[k],
					new_unsaturated_hydraulic_conductivity[k],
					new_moisture_content_total_water[k],
					new_moisture_content_free_water[k],
					parameters.hydraulic_properties,
					new_biomass_concentration_in_cell[k]);
		}
		double old_cell_average_substrate=0;
		if (substrate_old_values[0]>0)
			old_cell_average_substrate+=0.5*substrate_old_values[0];
		if (substrate_old_values[1]>0)
			old_cell_average_substrate+=0.5*substrate_old_values[1];


		double new_cell_average_moisture_content_free
		=0.5*new_moisture_content_free_water[0]+0.5*new_moisture_content_free_water[1];
		double old_cell_average_moisture_content_free
		=0.5*old_moisture_content_total_water[0]+0.5*old_moisture_content_total_water[1];
		double new_cell_average_moisture_content_total
		=0.5*new_moisture_content_total_water[0]+0.5*new_moisture_content_total_water[1];
		double old_cell_average_moisture_content_total
		=0.5*old_moisture_content_total_water[0]+0.5*old_moisture_content_total_water[1];
		double new_cell_average_biomass_concentration
		=0.5*new_biomass_concentration_in_cell[0]+0.5*new_biomass_concentration_in_cell[1];
		double old_cell_average_biomass_concentration
		=0.5*new_biomass_concentration_in_cell[0]+0.5*new_biomass_concentration_in_cell[1];
		double new_cell_average_unsaturated_hydraulic_conductivity
		=0.5*new_unsaturated_hydraulic_conductivity[0]+0.5*new_unsaturated_hydraulic_conductivity[1];
		double old_cell_average_unsaturated_hydraulic_conductivity
		=0.5*old_unsaturated_hydraulic_conductivity[0]+0.5*old_unsaturated_hydraulic_conductivity[1];

		double new_sink_factor=0;
		double old_sink_factor=0;
		if (test_transport==true)
		{
			new_sink_factor=0;
			old_sink_factor=0;
		}
		else if (parameters.homogeneous_decay_rate==true)
		{
			new_sink_factor=parameters.first_order_decay_factor;
			old_sink_factor=parameters.first_order_decay_factor;
		}
		else
		{
			/* *
			 * Some of the variables for the transport equation defined in the input file
			 * are provided in [mg_substrate/L_free_water]. They need to be transformed
			 * to [mg_substrate/cm3_soil] to be consistent with the primary variable in
			 * the transport equation and to [mg_biomass/cm3_soil] in case of the biomass
			 * concentration variable defined in the program as [mg_biomass/cm3_total_water].
			 * This is done in this way:
			 *
			 * half_velocity_constant[mg_substrate/L_free_water]
			 * =half_velocity_constant[mg_substrate/L_free_water]*
			 * moisture_content_free_water[cm3_free_water/cm3_soil]*
			 * free_water_volume_ratio [L_free_water/1000 cm3_free_water]
			 * =(1./1000.)*half_velocity_constant[mg_substrate/cm3_soil]
			 *
			 * biomass_concentration[mg_biomass/cm3_total_water]
			 * =biomass_concentration[mg_biomass/cm3_total_water]*
			 * moisture_content_total_water[cm3_total_water/cm3_soil]
			 * =biomass_concentration[mg_biomass/cm3_soil]
			 *
			 * sink_factor[1/s]
			 * =[mg_biomass/cm3_soil]*[mg_substrate/mg_biomass*s]
			 * /([mg_substrate/cm3_soil]
			 * */
			new_sink_factor
			=-1.*/*(new_cell_average_moisture_content_total/new_cell_average_moisture_content_free)*/
			new_cell_average_biomass_concentration*
			parameters.maximum_substrate_use_rate
			/(old_cell_average_substrate+(1./1000.)*parameters.half_velocity_constant);

			old_sink_factor
			=-1.*/*(old_cell_average_moisture_content_total/old_cell_average_moisture_content_free)*/
			old_cell_average_biomass_concentration*
			parameters.maximum_substrate_use_rate
			/(old_cell_average_substrate+(1./1000.)*parameters.half_velocity_constant);
		}

		double new_discharge_darcy_velocity=0.;
		double old_discharge_darcy_velocity=0.;
		if (test_transport==true)
		{
			new_discharge_darcy_velocity
			=parameters.richards_top_flow_value;

			old_discharge_darcy_velocity
			=parameters.richards_top_flow_value;
		}
		else
		{
			new_discharge_darcy_velocity
			=(1./new_cell_average_moisture_content_free)*
			new_cell_average_unsaturated_hydraulic_conductivity*
			((1./cell->diameter())*(new_pressure_values[1]-new_pressure_values[0])+1.);

			old_discharge_darcy_velocity
			=(1./old_cell_average_moisture_content_free)*
			old_cell_average_unsaturated_hydraulic_conductivity*
			((1./cell->diameter())*(old_pressure_values[1]-old_pressure_values[0])+1.);
		}

		double new_diffusion_value
		=parameters.dispersivity_longitudinal*fabs(new_discharge_darcy_velocity)+
		parameters.effective_diffusion_coefficient;

		double old_diffusion_value
		=parameters.dispersivity_longitudinal*fabs(old_discharge_darcy_velocity)+
		parameters.effective_diffusion_coefficient;

		double Peclet
		=0.5*cell->diameter()*new_discharge_darcy_velocity/new_diffusion_value;
		double beta
		=(1./tanh(Peclet) - 1./Peclet);
		double tau
		=0.5*cell->diameter()*beta/new_discharge_darcy_velocity;

		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		{
			for (unsigned int k=0; k<dofs_per_cell; ++k)
			{
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					for (unsigned int j=0; j<dofs_per_cell; ++j)
					{
						/*i=test function, j=concentration*/
						cell_mass_matrix_new(i,j)+=
								fe_values.shape_value(i,q_point)*
								fe_values.shape_value(k,q_point)*
								/*new_cell_average_moisture_content_free*/
								fe_values.shape_value(j,q_point)*
								fe_values.JxW(q_point)
								//							+
								//							(new_cell_average_moisture_content_free)*
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							new_cell_average_moisture_content_free*
								//							fe_values.shape_value(j,q_point)*
								//							fe_values.JxW(q_point)
								;

						cell_mass_matrix_old(i,j)+=
								fe_values.shape_value(i,q_point)*
								fe_values.shape_value(k,q_point)*
								/*old_cell_average_moisture_content_free*/
								fe_values.shape_value(j,q_point)*
								fe_values.JxW(q_point)
								//							+
								//							(old_cell_average_moisture_content_free)*
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							old_cell_average_moisture_content_free*
								//							fe_values.shape_value(j,q_point)*
								//							fe_values.JxW(q_point)
								;

						cell_laplace_matrix_new(i,j)+=
								/*Diffusive term*/
								fe_values.shape_grad(i,q_point)*

								/*new_cell_average_moisture_content_free*/
								fe_values.shape_value(k,q_point)*
								new_diffusion_value*
								fe_values.shape_grad(j,q_point)*
								fe_values.JxW(q_point)
								+
								/*Convective term*/
								fe_values.shape_value(i,q_point)*

								fe_values.shape_value(k,q_point)*
								advection_field*
								new_discharge_darcy_velocity*
								fe_values.shape_grad(j,q_point)*
								fe_values.JxW(q_point)
								//							+
								//							(new_cell_average_moisture_content_free)*
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(j,q_point)*
								//							fe_values.JxW(q_point)
								/*Reaction term*/
								-
								fe_values.shape_value(i,q_point)*

								/*new_cell_average_moisture_content_free*/
								fe_values.shape_value(k,q_point)*
								new_sink_factor*
								fe_values.shape_value(j,q_point)*
								fe_values.JxW(q_point)
								//							-
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							/*new_cell_average_moisture_content_free*/
								//							new_sink_factor*
								//							fe_values.shape_value(j,q_point)*
								//							fe_values.JxW(q_point)
								;

						cell_laplace_matrix_old(i,j)+=
								/*Diffusive term*/
								fe_values.shape_grad(i,q_point)*

								/*old_cell_average_moisture_content_free*/
								fe_values.shape_value(k,q_point)*
								old_diffusion_value*
								fe_values.shape_grad(j,q_point)*
								fe_values.JxW(q_point)
								+
								/*Convective term*/
								fe_values.shape_value(i,q_point)*

								fe_values.shape_value(k,q_point)*
								advection_field*
								old_discharge_darcy_velocity*
								fe_values.shape_grad(j,q_point)*
								fe_values.JxW(q_point)
								//							+
								//							(old_cell_average_moisture_content_free)*
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(j,q_point)*
								//							fe_values.JxW(q_point)
								/*Reaction term*/
								-
								fe_values.shape_value(i,q_point)*

								/*old_cell_average_moisture_content_free*/
								fe_values.shape_value(k,q_point)*
								old_sink_factor*
								fe_values.shape_value(j,q_point)*
								fe_values.JxW(q_point)
								//							-
								//							tau*
								//							advection_field*
								//							discharge_darcy_velocity*
								//							fe_values.shape_grad(i,q_point)*
								//
								//							/*old_cell_average_moisture_content_free*/
								//							old_sink_factor*
								//							fe_values.shape_value(j,q_point)*
								//							fe_values.JxW(q_point)
								;
					}

//					cell_rhs(i)-=
//							time_step*
//							(theta_transport)*
//							fe_values.shape_value(i,q_point)*
//
//							old_cell_average_substrate*
//							advection_field*
//							new_discharge_darcy_velocity*
//							(new_cell_average_moisture_content_free)*
//
//							(1./new_moisture_content_free_water[k])*
//							fe_values.shape_grad(k,q_point)*
//							fe_values.JxW(q_point)
//							+
//							time_step*
//							(1.-theta_transport)*
//							fe_values.shape_value(i,q_point)*
//
//							old_cell_average_substrate*
//							advection_field*
//							old_discharge_darcy_velocity*
//							(old_cell_average_moisture_content_free)*
//
//							(1./old_moisture_content_free_water[k])*
//							fe_values.shape_grad(k,q_point)*
//							fe_values.JxW(q_point);
				}
			}
		}

		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
		{
			face_boundary_indicator=cell->face(face)->boundary_indicator();

//			if ((parameters.transport_fixed_at_top==false) &&
//					(cell->face(face)->at_boundary()) &&
//					(face_boundary_indicator==0))
//			{
//				fe_face_values.reinit(cell,face);
//				double concentration_at_boundary
//				=(1./1000.)*parameters.transport_top_fixed_value;

//				for (unsigned int q_face_point=0; q_face_point<n_face_q_points; ++q_face_point)
//				{
//					for (unsigned int k=0; k<dofs_per_cell; k++)
//					{
//						for (unsigned int i=0; i<dofs_per_cell; ++i)
//						{
//							for (unsigned int j=0; j<dofs_per_cell; ++j)
//							{
//								cell_laplace_matrix_new(i,j)-=
//										fe_face_values.shape_value(i,q_face_point)*
//
//										fe_face_values.shape_value(k,q_face_point)*
//										discharge_darcy_velocity*
//										fe_face_values.shape_value(j,q_face_point)*
//										fe_face_values.JxW(q_face_point)
//										+
//										tau*
//										advection_field*
//										discharge_darcy_velocity*
//										fe_face_values.shape_grad(i,q_face_point)*
//
//										fe_face_values.shape_value(k,q_face_point)*
//										discharge_darcy_velocity*
//										fe_face_values.shape_value(j,q_face_point)*
//										fe_face_values.JxW(q_face_point)
//										;
//
//								cell_laplace_matrix_old(i,j)-=
//										fe_face_values.shape_value(i,q_face_point)*
//
//										fe_face_values.shape_value(k,q_face_point)*
//										discharge_darcy_velocity*
//										fe_face_values.shape_value(j,q_face_point)*
//										fe_face_values.JxW(q_face_point)
//										+
//										tau*
//										advection_field*
//										discharge_darcy_velocity*
//										fe_face_values.shape_grad(i,q_face_point)*
//
//										fe_face_values.shape_value(k,q_face_point)*
//										discharge_darcy_velocity*
//										fe_face_values.shape_value(j,q_face_point)*
//										fe_face_values.JxW(q_face_point)
//										;
//							}
//						}
//					}
//				}
//			}

			if ((parameters.transport_fixed_at_top==false) &&
					(cell->face(face)->at_boundary()) &&
					(face_boundary_indicator==1))
			{
				fe_face_values.reinit(cell,face);
				double concentration_at_boundary
				=(1./1000.)*parameters.transport_top_fixed_value;
				new_discharge_darcy_velocity
				=(1./new_cell_average_moisture_content_free)*
				parameters.richards_top_flow_value;
				old_discharge_darcy_velocity
				=(1./old_cell_average_moisture_content_free)*
				parameters.richards_top_flow_value;


				for (unsigned int q_face_point=0; q_face_point<n_face_q_points; ++q_face_point)
				{
					for (unsigned int i=0; i<dofs_per_cell; ++i)
					{
						for (unsigned int j=0; j<dofs_per_cell; ++j)
						{
							cell_laplace_matrix_new(i,j)+=
									fe_face_values.shape_value(i,q_face_point)*

									/*new_cell_average_moisture_content_free*/
									new_discharge_darcy_velocity*
									fe_face_values.shape_value(j,q_face_point)*
									fe_face_values.JxW(q_face_point)
//									+
//									(new_cell_average_moisture_content_free)*
//									tau*
//									advection_field*
//									discharge_darcy_velocity*
//									fe_face_values.shape_grad(i,q_face_point)*
//
//									new_cell_average_moisture_content_free*
//									discharge_darcy_velocity*
//									fe_face_values.shape_value(j,q_face_point)*
//									fe_face_values.JxW(q_face_point)
									;

							cell_laplace_matrix_old(i,j)+=
									fe_face_values.shape_value(i,q_face_point)*

									/*old_cell_average_moisture_content_free*/
									old_discharge_darcy_velocity*
									fe_face_values.shape_value(j,q_face_point)*
									fe_face_values.JxW(q_face_point)
//									+
//									(old_cell_average_moisture_content_free)*
//									tau*
//									advection_field*
//									discharge_darcy_velocity*
//									fe_face_values.shape_grad(i,q_face_point)*
//
//									old_cell_average_moisture_content_free*
//									discharge_darcy_velocity*
//									fe_face_values.shape_value(j,q_face_point)*
//									fe_face_values.JxW(q_face_point)
									;
						}

						cell_rhs(i)+=
								time_step*
								(theta_transport)*
								fe_face_values.shape_value(i,q_face_point)*

								/*new_cell_average_moisture_content_free*/
								concentration_at_boundary*
								new_discharge_darcy_velocity*
								fe_face_values.JxW(q_face_point)
//								+
//								time_step*
//								(theta_transport)*
//								(new_cell_average_moisture_content_free)*
//								tau*
//								advection_field*
//								discharge_darcy_velocity*
//								fe_face_values.shape_grad(i,q_face_point)*
//
//								new_cell_average_moisture_content_free*
//								concentration_at_boundary*
//								discharge_darcy_velocity*
//								fe_face_values.JxW(q_face_point)
								+
								time_step*
								(1.-theta_transport)*
								fe_face_values.shape_value(i,q_face_point)*

								/*old_cell_average_moisture_content_free*/
								concentration_at_boundary*
								old_discharge_darcy_velocity*
								fe_face_values.JxW(q_face_point)
//								+
//								time_step*
//								(1.-theta_transport)*
//								(old_cell_average_moisture_content_free)*
//								tau*
//								advection_field*
//								discharge_darcy_velocity*
//								fe_face_values.shape_grad(i,q_face_point)*
//
//								old_cell_average_moisture_content_free*
//								concentration_at_boundary*
//								discharge_darcy_velocity*
//								fe_face_values.JxW(q_face_point)
								;
					}
				}
			}
		}

    	cell_integer_index++;
    	cell->get_dof_indices(local_dof_indices);

    	for (unsigned int i=0; i<dofs_per_cell; ++i)
    	{
    		for (unsigned int j=0; j<dofs_per_cell; ++j)
    		{
    			laplace_matrix_new_transport
				.add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_new(i,j));
    			laplace_matrix_old_transport
				.add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_old(i,j));
    			mass_matrix_transport_new
				.add(local_dof_indices[i],local_dof_indices[j],cell_mass_matrix_new(i,j));
    			mass_matrix_transport_old
				.add(local_dof_indices[i],local_dof_indices[j],cell_mass_matrix_old(i,j));
    		}
    		system_rhs_transport(local_dof_indices[i])+=cell_rhs(i);
    	}
    }

    Vector<double> tmp(solution_transport.size ());

    mass_matrix_transport_old.vmult   ( tmp,old_solution_transport);
    system_rhs_transport.add          ( 1.0,tmp);
    laplace_matrix_old_transport.vmult( tmp,old_solution_transport);
    system_rhs_transport.add          (-(1-theta_transport)*time_step,tmp);

    system_matrix_transport.copy_from (mass_matrix_transport_new);
    system_matrix_transport.add       (theta_transport*time_step,laplace_matrix_new_transport);

    hanging_node_constraints.condense(system_matrix_transport);
    hanging_node_constraints.condense(system_rhs_transport);

    /* *
     * The boundary condition for the transport equation defined in the input file
     * is provided in [mg_substrate/L_free_water]. It needs to be transformed to
     * [mg_substrate/cm3_soil] to be consistent with the primary
     * variable in the transport equation. This is done in this way:
     *
     * boundary_condition[mg_substrate/L_free_water]
     * =boundary_condition[mg_substrate/L_free_water]*
     * moisture_content_free_water_at_boundary[cm3_free_water/cm3_soil]*
     * free_water_volume_ratio[L_free_water/1000 cm3_total_water]
     * =(1./1000.)*boundary_condition[mg_substrate/cm3_soil]
     * */
    if (parameters.transport_fixed_at_top==true)
    {
    	double boundary_pressure_at_top
		=VectorTools::point_value(dof_handler,
				solution_flow_old_iteration,
				Point<dim>(0.));
    	double boundary_specific_moisture_capacity=0;
    	double boundary_unsaturated_hydraulic_conductivity=0.;
    	double boundary_moisture_content_free_water=0.;
    	double boundary_moisture_content_total_water=0.;
    	double boundary_biomass_concentration
		=new_nodal_biomass_concentration[127];

    	hydraulic_properties(
    			boundary_pressure_at_top,
				boundary_specific_moisture_capacity,
				boundary_unsaturated_hydraulic_conductivity,
				boundary_moisture_content_total_water,
				boundary_moisture_content_free_water,
				parameters.hydraulic_properties,
				boundary_biomass_concentration);

    	double boundary_condition_substrate=0.;
    	if (test_transport==true)
    		boundary_condition_substrate
			=parameters.transport_top_fixed_value*(1./1000.);
    	else
    		boundary_condition_substrate
			=boundary_moisture_content_total_water*
			parameters.transport_top_fixed_value*(1./1000.);

    	std::map<unsigned int,double> boundary_values;
    	boundary_values.clear();
    	VectorTools::interpolate_boundary_values (dof_handler,
    			1,
				ConstantFunction<dim>(boundary_condition_substrate),
				boundary_values);
    	MatrixTools::apply_boundary_values (boundary_values,
    			system_matrix_transport,
				solution_transport,
				system_rhs_transport);
    }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_flow()
  {  
	  system_rhs_flow.reinit         (dof_handler.n_dofs());
	  system_matrix_flow.reinit      (sparsity_pattern);
	  mass_matrix_richards.reinit        (sparsity_pattern);
	  laplace_matrix_new_richards.reinit (sparsity_pattern);
	  laplace_matrix_old_richards.reinit (sparsity_pattern);

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
  	  //QuadratureSelector<dim-1> face_quadrature_formula(quadrature_option,order);
      QGauss<dim-1>     face_quadrature_formula(2);
	  FEValues<dim>fe_values(fe, quadrature_formula,
			  update_values|update_gradients|
			  update_JxW_values);
	  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
			  update_values|update_gradients|
			  update_normal_vectors|
			  update_quadrature_points|update_JxW_values);
	  const unsigned int dofs_per_cell  =fe.dofs_per_cell;
	  const unsigned int n_face_q_points=face_quadrature_formula.size();
	  const unsigned int n_q_points     =quadrature_formula.size();

	  FullMatrix<double> cell_mass_matrix       (dofs_per_cell,dofs_per_cell);
	  FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell,dofs_per_cell);
	  FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell,dofs_per_cell);
	  Vector<double>     cell_rhs               (dofs_per_cell);

	  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
	  Vector<double> old_pressure_values;
	  Vector<double> new_pressure_values;
	  double face_boundary_indicator;

	  unsigned int cell_integer_index=0;
	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell)
	  {
		  fe_values.reinit (cell);
		  cell_mass_matrix       =0;
		  cell_laplace_matrix_new=0;
		  cell_laplace_matrix_old=0;
		  cell_rhs               =0;

		  old_pressure_values.reinit(cell->get_fe().dofs_per_cell);
		  new_pressure_values.reinit(cell->get_fe().dofs_per_cell);

		  cell->get_dof_values(old_solution_flow,old_pressure_values);
		  cell->get_dof_values(solution_flow_old_iteration,new_pressure_values);
		  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		  {
			  for (unsigned int k=0; k<dofs_per_cell; k++)
			  {
				  double old_specific_moisture_capacity=0;
				  double old_unsaturated_hydraulic_conductivity=0;
				  double old_moisture_content_total_water=0;
				  double old_moisture_content_free_water=0;
				  double new_specific_moisture_capacity=0;
				  double new_unsaturated_hydraulic_conductivity=0;
				  double new_moisture_content_total_water=0;
				  double new_moisture_content_free_water=0;


				  double old_biomass_concentration_in_cell=0;
				  double new_biomass_concentration_in_cell=0;
				  if (transient_drying==false)
				  {
					  old_biomass_concentration_in_cell
					  =old_nodal_biomass_concentration[cell_integer_index+k];
					  new_biomass_concentration_in_cell
					  =new_nodal_biomass_concentration[cell_integer_index+k];
				  }

				  hydraulic_properties(
						  old_pressure_values[k],
						  old_specific_moisture_capacity,
						  old_unsaturated_hydraulic_conductivity,
						  old_moisture_content_total_water,
						  old_moisture_content_free_water,
						  parameters.hydraulic_properties,
						  old_biomass_concentration_in_cell);
				  hydraulic_properties(
						  new_pressure_values[k],
						  new_specific_moisture_capacity,
						  new_unsaturated_hydraulic_conductivity,
						  new_moisture_content_total_water,
						  new_moisture_content_free_water,
						  parameters.hydraulic_properties,
						  new_biomass_concentration_in_cell);

				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
					  for (unsigned int j=0; j<dofs_per_cell; ++j)
					  {
						  if (parameters.moisture_transport_equation.compare("head")==0)
						  {
							  cell_mass_matrix(i,j)+=
									  (theta_richards)*
									  new_specific_moisture_capacity*
									  fe_values.shape_value(k,q_point)*
									  fe_values.shape_value(i,q_point)*
									  fe_values.shape_value(j,q_point)*
									  fe_values.JxW(q_point)
									  +
									  (1-theta_richards)*
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
					  cell_rhs(i)-=
							  time_step*
							  (theta_richards)*
							  new_unsaturated_hydraulic_conductivity*
							  fe_values.shape_value(k,q_point)*
							  fe_values.shape_grad(i,q_point)[0]*
							  fe_values.JxW(q_point)
							  +
							  time_step*
							  (1.-theta_richards)*
							  old_unsaturated_hydraulic_conductivity*
							  fe_values.shape_value(k,q_point)*
							  fe_values.shape_grad(i,q_point)[0]*
							  fe_values.JxW(q_point);

					  if (parameters.moisture_transport_equation.compare("mixed")==0)
					  {
						  cell_rhs(i)-=(
								  new_moisture_content_total_water-
								  old_moisture_content_total_water)*
								  fe_values.shape_value(k,q_point)*
								  fe_values.shape_value(i,q_point)*
								  fe_values.JxW(q_point);
					  }
				  }
			  }
		  }

		  double flow=0.0;
		  if (transient_drying==false)
			  flow=parameters.richards_top_flow_value;

		  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
		  {
			  face_boundary_indicator=cell->face(face)->boundary_indicator();
			  if ((cell->face(face)->at_boundary()) &&
					  (face_boundary_indicator==1))
			  {
				  fe_face_values.reinit (cell,face);
				  for (unsigned int q_face_point=0; q_face_point<n_face_q_points; ++q_face_point)
					  for (unsigned int k=0; k<dofs_per_cell; ++k)
						  for (unsigned int i=0; i<dofs_per_cell; ++i)
							  cell_rhs(i)+=
									  time_step*
									  (theta_richards)*flow*
									  fe_face_values.shape_value(k,q_face_point)*
									  fe_face_values.shape_value(i,q_face_point)*
									  fe_face_values.JxW(q_face_point)
									  +
									  time_step*
									  (1.-theta_richards)*flow*
									  fe_face_values.shape_value(k,q_face_point)*
									  fe_face_values.shape_value(i,q_face_point)*
									  fe_face_values.JxW(q_face_point);
			  }
		  }
		  cell_integer_index++;
		  cell->get_dof_indices (local_dof_indices);
		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			  {
				  laplace_matrix_new_richards
				  .add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_new(i,j));
				  laplace_matrix_old_richards
				  .add(local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_old(i,j));
				  mass_matrix_richards
				  .add(local_dof_indices[i],local_dof_indices[j],cell_mass_matrix(i,j));
			  }
			  system_rhs_flow(local_dof_indices[i])+=cell_rhs(i);
		  }
	  }

	  Vector<double> tmp(solution_flow_new_iteration.size ());
	  if (parameters.moisture_transport_equation.compare("head")==0)
		  mass_matrix_richards.vmult(tmp,old_solution_flow);
	  else if (parameters.moisture_transport_equation.compare("mixed")==0)
		  mass_matrix_richards.vmult(tmp,solution_flow_old_iteration);
	  else
	  {
		  std::cout << "Moisture transport equation \""
				  << parameters.moisture_transport_equation
				  << "\" is not implemented. Error.\n";
		  throw -1;
	  }

	  system_rhs_flow.add          ( 1.0,tmp);
	  laplace_matrix_old_richards.vmult( tmp,old_solution_flow);
	  system_rhs_flow.add          (-(1-theta_richards)*time_step,tmp);

	  system_matrix_flow.copy_from(mass_matrix_richards);
	  system_matrix_flow.add      (theta_richards*time_step, laplace_matrix_new_richards);

	  hanging_node_constraints.condense(system_matrix_flow);
	  hanging_node_constraints.condense(system_rhs_flow);

	  std::map<unsigned int,double> boundary_values;
	  if (parameters.richards_fixed_at_bottom==true)
	  {
		  VectorTools::interpolate_boundary_values(dof_handler,
				  0,
				  ConstantFunction<dim>(parameters.richards_bottom_fixed_value),
				  boundary_values);
		  MatrixTools::apply_boundary_values (boundary_values,
				  system_matrix_flow,
				  solution_flow_new_iteration,
				  system_rhs_flow);
	  }
	  if (parameters.richards_fixed_at_top==true)
	  {
		  boundary_values.clear();
		  VectorTools::interpolate_boundary_values(dof_handler,
				  1,
				  ConstantFunction<dim>(parameters.richards_top_fixed_value),
				  boundary_values);
		  MatrixTools::apply_boundary_values(boundary_values,
				  system_matrix_flow,
				  solution_flow_new_iteration,
				  system_rhs_flow);
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::solve ()
  {
	  if(solve_flow && test_transport==false)
	  {
		  // Solve Richard's equation
		  SolverControl solver_control(100*solution_flow_new_iteration.size(),
				  1e-8*system_rhs_flow.l2_norm ());
		  SolverCG<> cg(solver_control);
		  PreconditionSSOR<> preconditioner;
		  preconditioner.initialize (system_matrix_flow, 1.2);
		  cg.solve(system_matrix_flow,solution_flow_new_iteration,
				  system_rhs_flow,preconditioner);
		  hanging_node_constraints.distribute(solution_flow_new_iteration);
	  }
	  if (transient_transport==true || test_transport==true)
	  {
		  // Solve Advection-Diffusion (substrate)
		  SolverControl solver_control_transport(1000*solution_transport.size(),
				  1e-8*system_rhs_transport.l2_norm());
		  SolverBicgstab<> bicgstab_transport(solver_control_transport);
		  PreconditionJacobi<> preconditioner_transport;
		  preconditioner_transport
		  .initialize(system_matrix_transport,1.0);
		  bicgstab_transport
		  .solve(system_matrix_transport,solution_transport,
				  system_rhs_transport,preconditioner_transport);
		  hanging_node_constraints.distribute(solution_transport);
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::output_results () const
  {
	  DataOut<dim> data_out;

	  data_out.attach_dof_handler(dof_handler);
	  data_out.add_data_vector(solution_flow_new_iteration,"pressure(cm_total_water)");
	  data_out.add_data_vector(solution_transport,"substrate(mg_substrate_per_cm3_soil)");
	  data_out.add_data_vector(new_nodal_biomass_concentration,"biomass(mg_biomass_per_cm3_total_water)");
	  data_out.add_data_vector(new_nodal_free_moisture_content,"free_water(cm3_free_water_per_cm3_soi)");
	  data_out.add_data_vector(new_nodal_total_moisture_content,"total_water(cm3_total_water_per_cm3_soil)");
	  data_out.add_data_vector(new_nodal_hydraulic_conductivity,"hydraulic_conductivity(cm_per_s)");
	  data_out.add_data_vector(new_nodal_specific_moisture_capacity,"specific_moisture_capacity(cm3_total_water_per_(cm3_soil)(cm_total_water))");
	  data_out.add_data_vector(new_nodal_flow_speed,"new_flow_speed(m3_free_water_per_s_per_cm2_soil)");
	  data_out.add_data_vector(old_nodal_flow_speed,"old_flow_speed(m3_free_water_per_s_per_cm2_soil)");
	  data_out.build_patches();

	  std::stringstream tsn;
	  tsn << timestep_number;
	  std::stringstream ts;
	  ts << time_step;
	  std::stringstream t;
	  //t << std::setprecision(1) << std::setw(6) << std::setfill('0') << std::fixed << ((time-milestone_time)/3600);
	  t << (int)(10*(time-milestone_time)/3600);
	  std::stringstream d;
	  d << dim;
	  std::stringstream c;
	  c << triangulation.n_active_cells();

	  std::string output_file_format=parameters.output_file_format;

	  std::string lm;
	  if (parameters.lumped_matrix==true)
		  lm="lumped_";

	  std::string time_period;
	  if (transient_drying==true)
		  time_period="drying";
	  else if (transient_saturation==true)
		  time_period="saturating";
	  else if (transient_transport==true)
		  time_period="transporting";

	  if (transient_transport==true && parameters.homogeneous_decay_rate==true)
		  time_period+="_decaying";

	  std::string filename;
	  if (test_transport==false)
	  {
		  filename = "solution_"
			  + parameters.moisture_transport_equation + "_" + lm
			  + d.str() + "d_"
//			  + "tsn_" + tsn.str()
//			  + "_c_" + c.str() + "_"
			  + time_period + "_t_" + t.str()
			  + output_file_format;
	  }
	  else
	  {
		  filename = "solution_"
				  + d.str() + "d_"
				  + "tsn_" + tsn.str()
				  + output_file_format;
	  }
	  std::ofstream output (filename.c_str());
	  if (output_file_format.compare(".gp")==0)
		  data_out.write_gnuplot (output);
	  else if (output_file_format.compare(".vtu")==0)
		  data_out.write_vtu (output);
	  else
	  {
		  std::cout << "Error in output function. Output file format "
				  << "not implemented.\n Current output file format is: "
				  << output_file_format << "\n" << "Options are: "
				  << ".gp, .vtu"
				  << std::endl;
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::initial_condition()
  {
	  /*
	   * Initial condition flow: Head [cm_total_water]
	   * */
	  VectorTools::project(dof_handler,
			  hanging_node_constraints,
			  QGauss<dim>(3),
			  ConstantFunction<dim> (parameters.initial_condition_homogeneous_flow),
			  old_solution_flow);
	  solution_flow_new_iteration
	  =old_solution_flow;
	  solution_flow_old_iteration
	  =old_solution_flow;
	  solution_flow_initial_time
	  =old_solution_flow;

	  /* *
	   * Initial condition transport: Concentration [mg_substrate/cm3_soil]
	   *
	   * The initial condition for the transport equation defined in the input file
	   * is given in mg_substrate/L_free_water. It needs to be transformed to
	   * mg_substrate/cm3_soil. This is done in this way:
	   *
	   * S[mg_substrate/cm3_soil]
	   * =moisture_content_free_water[cm3_free_water/cm3_soil]*S[mg_substrate/L_free_water]*
	   * water_volume_ratio[L_free_water/1000 cm3_free_water]
	   * =[cm3_free_water/cm3_soil]*[mg_substrate/L_free_water]*[L_free_water/1000 cm3_free_water]
	   * =(1/1000)*[mg_substrate/cm3_soil]
	   * */
	  double initial_pressure_values=parameters.initial_condition_homogeneous_flow;
	  double initial_specific_moisture_capacity=0;
	  double initial_unsaturated_hydraulic_conductivity=0.;
	  double initial_moisture_content_free_water=0.;
	  double initial_moisture_content_total_water=0.;
	  double initial_biomass_concentration=0.;

	  hydraulic_properties(
			  initial_pressure_values,
			  initial_specific_moisture_capacity,
			  initial_unsaturated_hydraulic_conductivity,
			  initial_moisture_content_total_water,
			  initial_moisture_content_free_water,
			  parameters.hydraulic_properties,
			  initial_biomass_concentration);

	  double initial_concentration_in_free_water
	  =parameters.initial_condition_homogeneous_transport;
	  double initial_concentration_in_soil
	  =(1./1000.)*initial_moisture_content_total_water//mg_substrate/cm3_free_water
	  /*initial_moisture_content_free_water*/
	  /*initial_concentration_in_free_water*/;

	  VectorTools::project (dof_handler,
			  hanging_node_constraints,
			  QGauss<dim>(3),
			  ConstantFunction<dim>(initial_concentration_in_soil),
			  old_solution_transport);
	  solution_transport=old_solution_transport;

	  /*
	   * Initial condition biomass: Concentration [mg_biomass/cm3_total_water]
	   *
	   * The initial condition for biomass defined in the input file is given in
	   * mg_biomass/L_total_water. It needs to be transformed to mg_biomass/cm3_total_water.
	   * This is done in this way:
	   *
	   * B[mg_biomass/L_total_water]
	   * =B[mg_biomass/L_total_water]*
	   * water_volume_ratio[L_total_water/1000 cm3_total_water]
	   * =[mg_biomass/L_total_water]*[L_total_water/1000 cm3_total_water]
	   * =(1/1000)*[mg_biomass/cm3_total_water]
	   * */

	   for (unsigned int i=0; i<dof_handler.n_dofs(); i++)
	   {
		   // mg_biomass/cm3_total_water
		   old_nodal_biomass_concentration[i]=0.;
		   new_nodal_biomass_concentration[i]=0.;
		   initial_nodal_biomass_concentration[i]=0.;
	   }

	   /*
	    * Velocities
	    * */
	   estimate_flow_velocities();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::run()
  {
	  read_grid();
	  setup_system();
	  initial_condition();

	  std::cout << "Solving problem with : "
			  << "\n\ttheta pressure     : " << theta_richards
			  << "\n\ttimestep_number_max: " << timestep_number_max
			  << "\n\ttime_step          : " << time_step
			  << "\n\ttime_max           : " << time_max
			  << "\n\trefinement_level   : " << refinement_level
			  << "\n\tuse_mesh_file      : " << use_mesh_file
			  << "\n\tmesh_filename      : " << mesh_filename
			  << "\n\tcells: " << triangulation.n_active_cells() << "\n";

	  unsigned int figure_count=0;
	  double total_initial_biomass=0.;
	  bool redefine_time_step=false;
	  for (timestep_number=1/*, time=time_step*/;
			  timestep_number<parameters.timestep_number_max;
			  ++timestep_number/*, time+=time_step*/)
	  {
		  double tolerance_flow=1000;
		  double tolerance_transport=0;
		  double mass_balance_ratio=0;
		  double transport_old_norm=0.;
		  double transport_new_norm=0.;
		  unsigned int iteration=0;
		  unsigned int step=0;
		  bool remain_in_loop=true;

		  do
		  {
			  /*
			   * We need to iterate at least two times (i'll do it 4 times)
			   * just to be safe) with each time step.
			   * So, if iteration==4 then the timestep is updated (by 10%)
			   * and iteration and tolerances are restarted.
			   * */
			  if (test_transport==false && iteration==3)
			  {
				  time_step=time_step/2.;

				  tolerance_flow=1000;
				  tolerance_transport=0;
				  mass_balance_ratio=0;
				  transport_old_norm=0.;
				  transport_new_norm=0.;
				  iteration=0;
			  }
			  /*
			   * The system is assembled using the defined time step:
			   * -Solve flow equation
			   * -Solve transport equation (if necessary)
			   * -Solve growth equation (if necessary)
			   * */
			  if (solve_flow && test_transport==false)
			  {
				  assemble_system_flow();
//				  solution_flow_old_iteration
//				  =solution_flow_new_iteration;

				  tolerance_flow
				  =fabs(solution_flow_new_iteration.l1_norm()-
						  solution_flow_old_iteration.l1_norm())
						  /solution_flow_new_iteration.l1_norm(); //(%)
			  }
			  if (transient_transport==true || test_transport==true)
			  {
				  transport_old_norm
				  =transport_new_norm;
				  assemble_system_transport();
				  transport_new_norm
				  =solution_transport.norm_sqr();
				  tolerance_transport
				  =fabs((transport_new_norm-transport_old_norm)/transport_new_norm);
			  }
			  /*
			   * Solve the system
			   */
			  solve();
			  solution_flow_old_iteration
			  =solution_flow_new_iteration;
			  /*
			   * With the newly calculated pressure solution, calculate the
			   * amount of moisture present in the domain and compare it with
			   * the mass entering or leaving the domain.
			   * */


			  if (test_transport==false)
			  {
				  calculate_mass_balance_ratio();

				  if (transient_transport==true && test_biomass_apearance==true)
					  total_initial_biomass
					  =change_in_biomass_from_beginning;

				  mass_balance_ratio
				  =change_in_moisture_from_beginning/
				  ((moisture_flow_at_inlet_cumulative+moisture_flow_at_inlet_current)
						  -(moisture_flow_at_outlet_cumulative+moisture_flow_at_outlet_current)
						  /*+total_initial_biomass*/);

				  estimate_flow_velocities();
			  }
			  /*
			   * Evaluate the condition to remain in the loop, also
			   * if the change in total moisture is negative (drying
			   * soil) AND is less than 1E-6, then override the mass
			   * balance error and activate the transport equation
			   * */
			  if (test_transport==false)
			  {
				  if (    tolerance_flow <1.5E-7 &&
						  tolerance_transport<1.5E-7 &&
						  /*(fabs(1.-mass_balance_ratio)<=1.E-6 || transient_transport==true)&&*/
						  iteration!=0)
					  remain_in_loop=false;
			  }
			  else
			  {
				  if (tolerance_transport<1.5E-7)
					  remain_in_loop=false;
			  }
			  /*
			   * Output some information about the current iteration.
			   */
			  //if (transient_transport==true && timestep_number>=28822)
//			  if (/*transient_transport==true &&*/ timestep_number>=5000)
			  if (transient_saturation==true && iteration>50 && iteration<=60)
			  {
				  std::cout.setf(std::ios::scientific,std::ios::floatfield);
				  std::cout.precision(10);
				  std::cout << "timestep_number: " << timestep_number
						  << "\tit: "    << iteration
						  << "\ttime: "  << std::setprecision(3) << std::setw(8) << std::setfill(' ') << std::fixed << (time+time_step-time_for_dry_conditions)/3600. << " s"
						  << "\tts: "    << std::setprecision(3) << std::setw(8) << std::setfill(' ') << std::fixed << time_step << " s"
						  << "\ttd: "    << transient_drying
						  << "\tts: "    << transient_saturation
						  << "\ttt: "    << transient_transport
						  << "\tT_tol: " << std::setprecision(2) << std::scientific << tolerance_transport
						  << "\tR_tol: " << std::setprecision(2) << std::scientific << tolerance_flow
						  << "\tmbr: "   << std::setprecision(8) << fabs(1.-mass_balance_ratio)
						  << "\tCFi: " << std::scientific << std::setprecision(4) << moisture_flow_at_inlet_current/time_step  << " m/s"
						  << "\tCFo: " << std::scientific << std::setprecision(4) << moisture_flow_at_outlet_current/time_step << " m/s"
						  << "\tCMi: " << std::scientific << std::setprecision(4) << moisture_flow_at_inlet_current     << " m"
						  << "\tTMi: " << std::fixed      << std::setprecision(4) << moisture_flow_at_inlet_cumulative  << " m"
						  << "\tCMo: " << std::scientific << std::setprecision(4) << moisture_flow_at_outlet_current    << " m"
						  << "\tTMo: " << std::fixed      << std::setprecision(4) << moisture_flow_at_outlet_cumulative << " m"
						  << "\tTb:  " << std::scientific << std::setprecision(4) << total_initial_biomass              << " m"
						  << "\tdTM: " << std::scientific << ((moisture_flow_at_inlet_cumulative+moisture_flow_at_inlet_current)
								  -(moisture_flow_at_outlet_cumulative+moisture_flow_at_outlet_current)
								  +total_initial_biomass) << " m"
						  << "\tdM: "  << std::fixed << std::setprecision(8) << change_in_moisture_from_beginning << " m"
						  << "\tdB: "  << std::fixed << change_in_biomass_from_beginning << " m"
						  << "\ttotal moisture: " << total_moisture << " m"
						  << "\ttotal biomass: " << total_biomass << " ?"
						  << "\n";
			  }
			  /*
			   * update the iteration
			   * */
			  iteration++;
			  step++;
		  }
		  while (remain_in_loop);

		  if(timestep_number==1)
			  time=time_step;
		  else
			  time=time+time_step;
		  /*
		   * Choose in which time period are we and note the time
		   */
		  double pressure_at_top
		  =VectorTools::point_value(dof_handler,
				  solution_flow_new_iteration,
				  Point<dim>(0.));
		  double biomass_concentration_at_top
		  =new_nodal_biomass_concentration[127];

		  if (test_transport==false)
		  {
			  if (transient_drying==true &&
					  fabs(1.-fabs(pressure_at_top)/
					  (parameters.domain_size+fabs(parameters.richards_bottom_fixed_value)))<3.1E-4/*7.5E-5*/)
			  {
				  transient_drying=false;
				  transient_saturation=true;
				  redefine_time_step=true;

				  time_for_dry_conditions=time;
				  milestone_time=time;
				  std::cout << "\tDry conditions reached at: "
						  << time_for_dry_conditions/3600 << " h\n"
						  << "\ttimestep_number: " << timestep_number << "\n"
						  << "\ttime_step: " << time_step << " s\n"
						  << "\tnumerical pressure at top: " << pressure_at_top << " m\n"
						  << "\texpected pressure at top: "
						  << (parameters.domain_size+fabs(parameters.richards_bottom_fixed_value)) << " m\n"
						  << "\tActivating moisture flow: "
						  << parameters.richards_top_flow_value << " cm/s\n"
						  << "\txxx: " << change_in_moisture_from_beginning << "\n";
			  }
			  if (transient_saturation==true &&
					  fabs(1.-fabs(moisture_flow_at_outlet_current/time_step)/
					  fabs(parameters.richards_top_flow_value))<2.6E-5/*1.1852E-5*/)
			  {
				  transient_saturation=false;
				  transient_transport=true;
				  for (unsigned int i=0; i<dof_handler.n_dofs(); i++)
					  new_nodal_biomass_concentration[i]=// mg_biomass/cm3_total_water
							  (1./1000.)*parameters.initial_condition_homogeneous_bacteria;

				  figure_count=0;
				  time_for_saturated_conditions=time;
				  milestone_time=time;
				  error_in_inlet_outlet_flow
				  =moisture_flow_at_inlet_current-moisture_flow_at_outlet_current;

				  std::cout << "\tSaturated conditions reached at: "
						  << time_for_saturated_conditions/3600 << " h\n"
						  << "\ttimestep_number: " << timestep_number << "\n"
						  << "\ttime_step: "       << time_step       << " s\n"
						  << "\tnumerical flow at top: "
						  << std::scientific << moisture_flow_at_inlet_current/time_step    << " m/s\n"
						  << "\talternative numerical flow at top: "
						  << std::scientific << alternative_current_flow_at_inlet/time_step << " m/s\n"
						  << "\tnumerical flow at bottom: "
						  << std::scientific << moisture_flow_at_outlet_current/time_step   << " m/s\n"
						  << "\texpected flow at bottom: "
						  << std::scientific << parameters.richards_top_flow_value          << " m/s\n"
						  << "\tError in inlet-outlet flow: "
						  << std::scientific << error_in_inlet_outlet_flow/time_step        << " m/s\n"
						  << "\tActivating nutrient flow: "
						  << std::scientific << parameters.transport_top_fixed_value        << " mg_substrate/m3_soil\n";
				  if (parameters.homogeneous_decay_rate==true)
					  std::cout << "Activating decay rate: "
					  << std::scientific << parameters.first_order_decay_factor << " 1/s\n";
			  }
		  }

		  /*
		   * Output info
		   */
		  if ((test_transport==false) && (
				  (timestep_number%100==0 && transient_transport==false) ||
				  (timestep_number%100==0 && transient_saturation==true) ||
				  (timestep_number%1000==0 && transient_transport==true)))
		  {
			  std::cout.setf(std::ios::fixed,std::ios::floatfield);
			  std::setprecision(10);
			  std::cout << "Time step " << timestep_number
					  << "\ttime: "     << (time-milestone_time)/3600;

			  if (transient_drying==true)
				  std::cout << "\tdrying";
			  else if (transient_saturation)
				  std::cout << "\tsaturation";
			  else
				  std::cout << "\ttransport";

			  if (transient_drying==true)
				  std::cout << "\tX: " << fabs(1.-fabs(pressure_at_top)/
						  (parameters.domain_size+fabs(parameters.richards_bottom_fixed_value)));
			  else if (transient_saturation)
				  std::cout << "\tX: " << fabs(1.-fabs(moisture_flow_at_outlet_current/time_step)/
						  fabs(parameters.richards_top_flow_value));

			  std::cout << "\tts: "       << time_step
					  << "\tit: "       << step
					  << "\tmbr: "      << fabs(1.-mass_balance_ratio);

			  std::cout << " = 1.-" << std::scientific << std::setprecision(8) << change_in_moisture_from_beginning << "/[("
					  << std::scientific << std::setprecision(8) << moisture_flow_at_inlet_cumulative  << "+"
					  << std::scientific << std::setprecision(8) << moisture_flow_at_inlet_current  << ")-("
					  << std::scientific << std::setprecision(8) << moisture_flow_at_outlet_cumulative << "+"
					  << std::scientific << std::setprecision(8) << moisture_flow_at_outlet_current << ")]"

					  << " = 1.-" << std::scientific << std::setprecision(8) << change_in_moisture_from_beginning << "/["
					  << std::scientific << std::setprecision(8) << (moisture_flow_at_inlet_cumulative+moisture_flow_at_inlet_current)
					  - (moisture_flow_at_outlet_cumulative + moisture_flow_at_outlet_current) << "]"

					  << "\t" << std::scientific << alternative_current_flow_at_inlet/time_step << " m/s"
					  << "\t" << std::scientific << alternative_cumulative_flow_at_inlet << " m"

					  << "\tCFi: " << std::scientific << moisture_flow_at_inlet_current/time_step  << " m/s"
					  << "\tCFo: " << std::scientific << moisture_flow_at_outlet_current/time_step << " m/s"
					  << "\tTMi: " << std::fixed << moisture_flow_at_inlet_cumulative  << " m"
					  << "\tTMo: " << std::fixed << moisture_flow_at_outlet_cumulative << " m"
					  << "\tdB: "  << std::fixed << change_in_biomass_from_beginning   << " m"
					  << "\ttotal moisture: " << total_moisture << " m"
					  << "\ttotal biomass: "  << total_biomass  << " ?";

			  std::cout << std::endl;
		  }
		  else if (test_transport==true)
		  {
			  std::cout << "Time step " << timestep_number << "\tts: " << time_step << "\n";
		  }
		  /*
		   * Output solution files
		   */
		  if ((test_transport==false) && (
				  //(transient_saturation==true && time-milestone_time>=figure_count*360  && (time-milestone_time)/3600<1.1) ||
				  (transient_saturation==true && time-milestone_time>=figure_count*1800) ||
				  (transient_transport==true && time-milestone_time>=figure_count*1800/*21600*/ /*&& (time-milestone_time)/3600<1130*/)
				  //(transient_transport==true && time-milestone_time>=figure_count*1800 && (time-milestone_time)/3600>=1130)
				  ))
		  {

			  output_results();
			  figure_count++;
		  }
		  else if (test_transport==true)
		  {
			  output_results();
			  figure_count++;
		  }


		  /*
		   * Update timestep
		   */
		  if (test_transport==false)
		  {
			  if (redefine_time_step==true)
			  {
				  time_step=1;
				  redefine_time_step=false;
			  }
			  else if (step<15)
			  {
				  if (transient_drying==true || transient_transport==true)
					  time_step=time_step*2;
				  else
					  time_step=time_step*2;
			  }

			  if (time_step<1)
				  time_step=1;

			  if (time_step>3600 && transient_drying==true)
				  time_step=3600;
			  else if (time_step>60 && transient_transport==true)
				  time_step=60;
			  else if (time_step>30 && transient_saturation==true)
				  time_step=30;
		  }
		  /*
		   * Update solutions
		   */
		  old_solution_flow
		  =solution_flow_new_iteration;
		  old_solution_transport
		  =solution_transport;
		  moisture_flow_at_inlet_cumulative
		  +=moisture_flow_at_inlet_current;
		  moisture_flow_at_outlet_cumulative
		  +=moisture_flow_at_outlet_current+error_in_inlet_outlet_flow;

		   for (unsigned int i=0; i<old_nodal_biomass_concentration.size(); i++)
			   old_nodal_biomass_concentration[i]=
					   new_nodal_biomass_concentration[i]; // mg_biomass/m3
	  }
/*
 * Save final states to be used as initial conditions for further analyses
 * */
	  output_results();
	  std::ofstream file;

	  open_file(file,"final_state_flow.ph");
	  solution_flow_new_iteration.block_write(file);
	  close_file(file);

	  open_file(file,"final_state_transport.ph");
	  solution_transport.block_write(file);
	  close_file(file);

	  open_file(file,"final_state_bacteria.ph");
	  close_file(file);

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

			clock_t t1,t2;
			t1=clock();
  			deallog.depth_console (0);
			Heat_Pipe<1> laplace_problem(argc,argv);
			laplace_problem.run();
			t2=clock();

			float time_diff
			=((float)t2-(float)t1);

			std::cout << time_diff << std::endl;
			std::cout << time_diff/CLOCKS_PER_SEC << std::endl;
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
