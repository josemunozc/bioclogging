/*
  Program to generate results for Chapter 8.
  The program generates results
  
  Using different boundary conditions:
  ++ Turbulent
  ++ Non-turbulent
  ++ With vegetation

  Using different initial conditions:
  ++ Analytical
  ++ Homogeneous
  ++ Experimental
  
  Using different meteorological data:
  ++ Analytical
  ++ Measured on site
  ++ Obtained from Met Office
*/

#include <deal.II/base/quadrature_lib.h>
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
    //--- Temperature functions ---
    void read_grid_temperature();
    void setup_system_temperature();
    void assemble_system_temperature(const double previous_new_temperature,
			 const double previous_new_pressure);
    void assemble_system_interval_temperature(const typename DoFHandler<dim>::active_cell_iterator &begin,
					      const typename DoFHandler<dim>::active_cell_iterator &end,
					      const double previous_new_temperature,
					      const double previous_new_pressure);
    void solve_temperature();
    void initial_condition_temperature();
    //--- Pressure functions ---
    void read_grid_pressure();
    void setup_system_pressure();
    void assemble_system_pressure(const double previous_new_temperature,
				  const double previous_new_pressure);
    void assemble_system_interval_pressure(const typename DoFHandler<dim>::active_cell_iterator &begin,
					   const typename DoFHandler<dim>::active_cell_iterator &end,
					   const double previous_new_temperature,
					   const double previous_new_pressure);
    void solve_pressure  ();
    
    void output_results () const; 
    void fill_output_vectors();
    void update_met_data ();

    Triangulation<dim>   triangulation;
    Triangulation<dim>   triangulation_pressure;
    DoFHandler<dim>      dof_handler;
    DoFHandler<dim>      dof_handler_pressure;
    FE_Q<dim>            fe;
    FE_Q<dim>            fe_pressure;
    
    ConstraintMatrix     hanging_node_constraints,hanging_node_constraints_pressure;
    
    SparsityPattern      sparsity_pattern,sparsity_pattern_pressure;
    
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix_new;
    SparseMatrix<double> laplace_matrix_old;
    Vector<double>       system_rhs;
    Vector<double>       solution;
    Vector<double>       old_solution;

    SparseMatrix<double> system_matrix_pressure;
    SparseMatrix<double> mass_matrix_pressure;
    SparseMatrix<double> laplace_matrix_new_pressure;
    SparseMatrix<double> laplace_matrix_old_pressure;
    Vector<double>       system_rhs_pressure;
    Vector<double>       solution_pressure;
    Vector<double>       old_solution_pressure;
    
    unsigned int timestep_number_max;
    unsigned int timestep_number;
    double       time;
    double       time_step;
    double       time_max;
    double       theta_temperature;
    double       theta_pressure;
    double       initial_time;
    
    Threads::ThreadMutex assembler_lock;
    
    const bool analytic;
    bool fixed_bc_at_bottom;
    const std::string surface_type;
    std::string author;
    std::string met_data_type;
    std::string filename_preffix;
    
    const int  preheating_step;
    
    std::vector< std::vector<int> >    date_and_time;
    std::vector< std::vector<double> > met_data;
    
    double new_air_temperature;
    double new_relative_humidity;
    double new_wind_speed;
    double new_wind_direction;
    double new_solar_radiation;
    double new_precipitation;
    
    double old_air_temperature;
    double old_relative_humidity;
    double old_wind_speed;
    double old_wind_direction;
    double old_solar_radiation;
    double old_precipitation;
    
    double canopy_density_period_1;
    double canopy_density_period_2;
    double canopy_density;

    const bool moisture_movement;
    double hydraulic_constant_b;
    double saturated_moisture_content;
    double saturated_hydraulic_conductivity;
    double saturated_water_pressure;
    double wilting_water_pressure;
    bool wilting_point_reached;
    bool saturation_point_reached;
    
    std::vector< std::vector<double> > soil_bha_temperature;
    std::vector< std::vector<double> > soil_bha_pressure;
    std::vector< std::vector<double> > soil_heat_fluxes;
    std::vector< std::vector<double> > soil_bha_thermal_conductivity;
    std::vector< std::vector<double> > soil_bha_thermal_heat_capacity;
    std::vector< std::vector<double> > soil_bha_density;
    std::vector< std::vector<double> > soil_bha_moisture_content;
    std::vector< std::vector<double> > soil_bha_hydraulic_conductivity;
    std::vector< std::vector<double> > soil_bha_hydraulic_moisture_capacity;

    Point<dim> borehole_A_depths[35];
    
    DeclException2 (Mismatch, double, double,
		    << "Geometry mismatch!!" << "\n"
		    << "value 1: " << arg1 << "\n"
		    << "value 2: " << arg2);

    Parameters::AllParameters<dim>  parameters;
  };
  /*
    Number of lines in met_file (and timesteps)
    --ph_1 (13 years)  ....  113841 - 1 ... 113840
  */
  template<int dim>
  Heat_Pipe<dim>::Heat_Pipe(int argc, char *argv[])
    :
    dof_handler(triangulation),
    dof_handler_pressure(triangulation_pressure),
    fe(1),
    fe_pressure(1),
    initial_time(0.),
    analytic(false),
    surface_type("soil"),
    preheating_step(1),
    moisture_movement(false),
    hydraulic_constant_b(11.4),
    saturated_moisture_content(0.4),
    saturated_hydraulic_conductivity(1.3e-6),
    saturated_water_pressure(-0.405),
    wilting_water_pressure(-150.)
  {
	  std::cout << "Program run with the following arguments:\n";
	  for (int i=0; i<argc; i++)
	  {
		  std::cout << "arg " << i << " : " << argv[i] << "\n";
	  }


    std::string input_filename = argv[6];
    std::cout << "parameter file: " << input_filename << "\n";
    
    ParameterHandler prm;
    Parameters::AllParameters<dim>::declare_parameters (prm);
    prm.read_input (input_filename);
    parameters.parse_parameters (prm);
    
    theta_temperature   = parameters.theta;
    theta_pressure      = parameters.theta;
    timestep_number_max = parameters.timestep_number_max;
    time_step           = parameters.time_step;
    time_max            = time_step*timestep_number_max;

    std::cout << "Solving problem with : \n"
	      << "\ttheta temperature  : " << theta_temperature << "\n"
	      << "\ttheta pressure     : " << theta_pressure << "\n"
	      << "\ttimestep_number_max: " << timestep_number_max << "\n"
	      << "\ttime_step          : " << time_step << "\n"
	      << "\ttime_max           : " << time_max  << "\n";

    
    if (argc==4 ||
	argc==7)
      {
	author=argv[1];
	met_data_type=argv[2];

	if ((met_data_type!="trl_met_data")&&
	    (met_data_type!="met_office_data"))
	  {
	    std::cout << "wrong met data type\n";
	    throw -1;
	  }

	std::string bottom_bc=argv[3];
	std::string bottom_bc_false="false";
	std::string bottom_bc_true="true";
	if (bottom_bc==bottom_bc_false)
	  fixed_bc_at_bottom=false;
	else if (bottom_bc==bottom_bc_true)
	  fixed_bc_at_bottom=true;
	else
	  {
	    std::cout << "wrong bottom_bc\n";
	    throw -1;
	  }

	if (argc>4)
	  {
	    canopy_density_period_1=atof(argv[4]);
	    canopy_density_period_2=atof(argv[5]);
	  }
	else
	  {
	    canopy_density_period_1=0.;
	    canopy_density_period_2=0.;
	  }

	if ((canopy_density_period_1<0.)||(canopy_density_period_1>1.0)||
	    (canopy_density_period_2<0.)||(canopy_density_period_2>1.0))
	  {
	    std::cout << "wrong canopy_density\n";
	    throw -1;
	  }
      }
    else
      {
	std::cout << "Not enough parameters\n"
		  << "prog_name\tauthor_name\ttrl_met_data\tbottom_bc\n"
		  << "or\n"
		  << "prog_name\tauthor_name\ttrl_met_data\tbottom_bc\tcanopy_density_period_1\tcanopy_density_period_2\n";
	throw -1;
      }

    filename_preffix
      =author+"_ph1_1d_wos_";
    if (fixed_bc_at_bottom==true)
      filename_preffix
	+="fx";
    else
      filename_preffix
	+="fr";


    std::cout << "Solving problem with the following data:\n"
	      << "\tAuthor: " << author << "\n"
	      << "\tbottom_bc: ";
    if (fixed_bc_at_bottom==true)
      std::cout << "true\n";
    else
      std::cout << "false\n";
    std::cout << "\tcanopy density period 1: " << canopy_density_period_1 << "\n"
	      << "\tcanopy density period 2: " << canopy_density_period_2 << "\n";


    for (unsigned int i=0; i<(sizeof borehole_A_depths)/(sizeof borehole_A_depths[0]); i++)
      borehole_A_depths[i] = Tensor<1,dim,double>();
    Names names(preheating_step,
		met_data_type);
    if ((sizeof borehole_A_depths)/(sizeof borehole_A_depths[0])<names.road_depths.size())
      {
	std::cout << "Not enough points for borehole depths definition" << std::endl;
        throw 1;
      }

    for (unsigned int i=0; i<names.road_depths.size(); i++)
      borehole_A_depths[i][0] = -1.*names.road_depths[i];

    timestep_number=0;
    time=0;

    new_air_temperature=0;
    new_relative_humidity=0;
    new_wind_speed=0;
    new_wind_direction=0;
    new_solar_radiation=0;
    new_precipitation=0;

    old_air_temperature=0;
    old_relative_humidity=0;
    old_wind_speed=0;
    old_wind_direction=0;
    old_solar_radiation=0;
    old_precipitation=0;

    canopy_density=0.;
    wilting_point_reached=false;
    saturation_point_reached=false;
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  Heat_Pipe<dim>::~Heat_Pipe ()
  {
    dof_handler.clear ();
    dof_handler_pressure.clear ();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::read_grid_temperature()
  {
    GridGenerator::hyper_cube (triangulation,-14, 0);
    triangulation.refine_global (10);
    dof_handler.distribute_dofs (fe);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::read_grid_pressure()
  {
    GridGenerator::hyper_cube (triangulation_pressure,-14, 0);
    triangulation_pressure.refine_global (10);
    dof_handler_pressure.distribute_dofs (fe_pressure);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::setup_system_temperature()
  {
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
  					     hanging_node_constraints);
    hanging_node_constraints.close ();

    CompressedSparsityPattern csp  (dof_handler.n_dofs(),
  				    dof_handler.n_dofs());
    
    DoFTools::make_sparsity_pattern (dof_handler, csp);
        
    hanging_node_constraints.condense (csp);    
    sparsity_pattern.copy_from (csp);    
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
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_temperature(const double previous_new_temperature,
						   const double previous_new_pressure)
  {  
    system_rhs.reinit         (dof_handler.n_dofs());
    system_matrix.reinit      (sparsity_pattern);
    mass_matrix.reinit        (sparsity_pattern);
    laplace_matrix_new.reinit (sparsity_pattern);
    laplace_matrix_old.reinit (sparsity_pattern);

    Vector<double> tmp      (solution.size ());
    
    const unsigned int n_threads = multithread_info.n_threads();
    Threads::ThreadGroup<> threads;
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
    std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
      thread_ranges
      = Threads::split_range<active_cell_iterator> (dof_handler.begin_active (),
  						    dof_handler.end (),
  						    n_threads);
    for (unsigned int thread=0; thread<n_threads; ++thread)
      threads += Threads::new_thread (&Heat_Pipe::assemble_system_interval_temperature,
  				      *this,
  				      thread_ranges[thread].first,
  				      thread_ranges[thread].second,
  				      previous_new_temperature,
				      previous_new_pressure);
    threads.join_all ();
    
    mass_matrix.vmult        ( tmp,old_solution);
    system_rhs.add           ( 1.0,tmp);
    laplace_matrix_old.vmult ( tmp,old_solution);
    system_rhs.add           (-(1 - theta_temperature) * time_step,tmp);

    system_matrix.copy_from (mass_matrix);
    system_matrix.add       (theta_temperature * time_step, laplace_matrix_new);

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);
    
    if (fixed_bc_at_bottom)
      {
	std::map<unsigned int,double> boundary_values;
	VectorTools::interpolate_boundary_values (dof_handler,
						  0,
						  ConstantFunction<dim>(10.95),
						  boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
					    system_matrix,
					    solution,
					    system_rhs);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_interval_temperature(const typename DoFHandler<dim>::active_cell_iterator &begin,
							    const typename DoFHandler<dim>::active_cell_iterator &end,
							    const double previous_new_temperature,
							    const double previous_new_pressure)
  {
    QGauss<dim>       quadrature_formula(3);
    QGauss<dim-1>     face_quadrature_formula (3);
  
    FEValues<dim>     fe_values      (fe, quadrature_formula,
  				      update_values | update_gradients |
  				      update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
  				      update_values | update_gradients |
  				      update_quadrature_points | update_JxW_values);
  
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size ();
  
    FullMatrix<double> cell_mass_matrix        (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_new (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_old (dofs_per_cell,dofs_per_cell);
    Vector<double>     cell_rhs                (dofs_per_cell);  
  
    std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);
    std::vector<double> old_pressure_values(n_q_points);
    std::vector<double> new_pressure_values(n_q_points);    
    double face_boundary_indicator;
    
    typename DoFHandler<dim>::active_cell_iterator cell;
    for (cell=begin; cell!=end; ++cell)
      {
  	fe_values.reinit (cell);
  	cell_mass_matrix        = 0;
  	cell_laplace_matrix_new = 0;
  	cell_laplace_matrix_old = 0;
  	cell_rhs                = 0;
	
 	if (moisture_movement==true)
	  {
	    fe_values.get_function_values(old_solution_pressure,
					  old_pressure_values);
	    fe_values.get_function_values(solution_pressure,
					  new_pressure_values);
	  }
	
  	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	  {
	    double average_pressure=0.;
	    double moisture_content=0.23;
	    if (moisture_movement==true)
	      {
		average_pressure
		  =(0.5*old_pressure_values[q_point]+
		    0.5*new_pressure_values[q_point]);
		if (average_pressure>=-10.)
		  average_pressure=-10.;
		if (average_pressure<=wilting_water_pressure)
		  average_pressure=wilting_water_pressure;
		
		moisture_content
		  =saturated_moisture_content
		  *pow(average_pressure
		       /saturated_water_pressure,(-1./hydraulic_constant_b));
		
		if (!(moisture_content==moisture_content) ||
		    (moisture_content>saturated_moisture_content) ||
		    (moisture_content<0.00))
		  {
		    std::cout << "\t\tError in MaterialData: Moisture content out of range\n"
			      << "\t\tMoisture content: " << std::scientific << moisture_content 
			      << " is out of range: " << 0.0 << "---" << saturated_moisture_content << std::endl
			      << "\t\tcell center: " << cell->center() << std::endl
			      << "\t\tprevious new temperature: " << previous_new_temperature << std::endl
			      << "\t\told pressure value" << old_pressure_values[q_point] << std::endl
			      << "\t\tnew pressure value" << new_pressure_values[q_point] << std::endl
			      << "\t\taverage_pressure: " << average_pressure << "\t\tmoisture_content: " << moisture_content << std::endl
			      << "\t\tsaturation point reached: " << saturation_point_reached << std::endl
			      << "\t\twilting point reached: " << wilting_point_reached << std::endl;
		    throw -1;
		  }
	      }
	    
	    MaterialData material_data (dim,false,moisture_content,moisture_movement);
	    double thermal_conductivity  = material_data.get_soil_thermal_conductivity(3);
	    double thermal_heat_capacity = material_data.get_soil_heat_capacity(3);
	    double density               = material_data.get_soil_density(3);

	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		{
		  cell_mass_matrix(i,j)       +=(thermal_heat_capacity * density *
						 fe_values.shape_value (i,q_point) *
						 fe_values.shape_value (j,q_point) *
						 fe_values.JxW (q_point));
		  cell_laplace_matrix_new(i,j)+=(thermal_conductivity *
						 fe_values.shape_grad (i, q_point) *
						 fe_values.shape_grad (j, q_point) *
						 fe_values.JxW (q_point));
		  cell_laplace_matrix_old(i,j)+=(thermal_conductivity *
						 fe_values.shape_grad (i, q_point) *
						 fe_values.shape_grad (j, q_point) *
						 fe_values.JxW (q_point));
		}
	  }
	
  	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)	
  	  {                                                                               
  	    face_boundary_indicator = cell->face(face)->boundary_indicator();
  	    if ((cell->face(face)->at_boundary()) &&
  		(face_boundary_indicator==1)) 
  	      {
  		fe_face_values.reinit (cell,face);
		
  		double old_surface_temperature
  		  =VectorTools::point_value(dof_handler,
					    old_solution,
					    Point<dim>(0.));
		double new_surface_temperature
  		  =previous_new_temperature;
		
		double old_surface_pressure=0.;
		double new_surface_pressure=0.;
		if (moisture_movement==true)
		  {
		    old_surface_pressure
		      =VectorTools::point_value(dof_handler_pressure,
						old_solution_pressure,
						Point<dim>(0.));
		    new_surface_pressure
		      =previous_new_pressure;
		  }
  		BoundaryConditions boundary_condition_old (analytic,
  							   time_step*(timestep_number-1) + initial_time,
  							   old_air_temperature,
  							   old_solar_radiation,
  							   old_wind_speed,
  							   old_relative_humidity,
  							   old_precipitation,
  							   old_surface_temperature,
							   old_surface_pressure,
							   moisture_movement);
  		BoundaryConditions boundary_condition_new (analytic,
  							   time_step*timestep_number + initial_time,
  							   new_air_temperature,
  							   new_solar_radiation,
  							   new_wind_speed,
  							   new_relative_humidity,
  							   new_precipitation,
  							   new_surface_temperature,
							   new_surface_pressure,
							   moisture_movement);
  		double shading_factor=0.;
  		double new_canopy_temperature=0.;
  		double old_canopy_temperature=0.;
  		if (author=="Best")
  		  {
  		    new_canopy_temperature =
  		      boundary_condition_new
  		      .get_canopy_temperature (surface_type,
  					       author,
  					       canopy_density);
  		    old_canopy_temperature =
  		      boundary_condition_old
  		      .get_canopy_temperature (surface_type,
  					       author,
  					       canopy_density);
  		  }
		
  		double outbound_convective_coefficient_new =
  		  boundary_condition_new
  		  .get_outbound_coefficient(surface_type,
  					    author,
  					    canopy_density,
  					    old_surface_temperature,
  					    new_surface_temperature,
  					    true);
  		double outbound_convective_coefficient_old =
  		  boundary_condition_old
  		  .get_outbound_coefficient(surface_type,
  					    author,
  					    canopy_density,
  					    old_surface_temperature,
  					    new_surface_temperature,
  					    false);
  		double inbound_heat_flux_new =
  		  boundary_condition_new
  		  .get_inbound_heat_flux(surface_type,
  					 author,
  					 shading_factor,
  					 new_canopy_temperature,
  					 canopy_density,
  					 old_surface_temperature,
  					 new_surface_temperature,
  					 true);
  		double inbound_heat_flux_old =
  		  boundary_condition_old
  		  .get_inbound_heat_flux(surface_type,
  					 author,
  					 shading_factor,
  					 old_canopy_temperature,
  					 canopy_density,
  					 old_surface_temperature,
  					 new_surface_temperature,
  					 false);
		

  		/*
  		  Save the calculated heat fluxes.                                                                                                                  
  		*/
  		{
  		  double solar_heat_flux;
  		  double inbound_convective_heat_flux;
  		  double inbound_evaporative_heat_flux;
  		  double inbound_infrared_heat_flux;
  		  double outbound_convective_coefficient;
  		  double outbound_evaporative_coefficient;
  		  double outbound_infrared_coefficient;
  		  boundary_condition_new.print_inbound_heat_fluxes (solar_heat_flux,
  								    inbound_convective_heat_flux,
  								    inbound_evaporative_heat_flux,
  								    inbound_infrared_heat_flux,
  								    outbound_convective_coefficient,
  								    outbound_infrared_coefficient,
  								    outbound_evaporative_coefficient);

  		  soil_heat_fluxes[timestep_number-1][0] = solar_heat_flux;
  		  soil_heat_fluxes[timestep_number-1][1] = inbound_convective_heat_flux;
  		  soil_heat_fluxes[timestep_number-1][2] = inbound_evaporative_heat_flux;
  		  soil_heat_fluxes[timestep_number-1][3] = inbound_infrared_heat_flux;
  		  soil_heat_fluxes[timestep_number-1][4] = outbound_convective_coefficient*new_surface_temperature;
  		  soil_heat_fluxes[timestep_number-1][5] = outbound_infrared_coefficient*new_surface_temperature;
  		  soil_heat_fluxes[timestep_number-1][6] = outbound_convective_coefficient;
  		  soil_heat_fluxes[timestep_number-1][7] = outbound_infrared_coefficient;
  		  soil_heat_fluxes[timestep_number-1][8] = outbound_convective_coefficient_new*new_surface_temperature;
  		  soil_heat_fluxes[timestep_number-1][9] = inbound_heat_flux_new;
  		}		  

  		for (unsigned int q_face_point=0; q_face_point<n_face_q_points; ++q_face_point)
  		  for (unsigned int i=0; i<dofs_per_cell; ++i)
  		    {
  		      for (unsigned int j=0; j<dofs_per_cell; ++j)
  			{
  			  cell_laplace_matrix_new(i,j)+=(outbound_convective_coefficient_new *
  							 fe_face_values.shape_value (i,q_face_point) *
  							 fe_face_values.shape_value (j,q_face_point) *
  							 fe_face_values.JxW         (q_face_point));
  			  cell_laplace_matrix_old(i,j)+=(outbound_convective_coefficient_old *
  							 fe_face_values.shape_value (i,q_face_point) *
							 fe_face_values.shape_value (j,q_face_point) *
  							 fe_face_values.JxW         (q_face_point));
  	 		}
  		      cell_rhs(i)+=((inbound_heat_flux_new *
  				     time_step * theta_temperature *
  				     fe_face_values.shape_value (i,q_face_point) *
  				     fe_face_values.JxW (q_face_point))
  				    +
  				    (inbound_heat_flux_old *
  				     time_step * (1-theta_temperature) *
  				     fe_face_values.shape_value (i,q_face_point) *
  				     fe_face_values.JxW (q_face_point)));
  	 	    }
  	      }
  	  }
	
  	cell->get_dof_indices (local_dof_indices);
	
  	assembler_lock.acquire ();
	
  	for (unsigned int i=0; i<dofs_per_cell; ++i)
  	  {
  	    for (unsigned int j=0; j<dofs_per_cell; ++j)
  	      {
  		laplace_matrix_new.add (local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_new(i,j));
  		laplace_matrix_old.add (local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_old(i,j));
  		mass_matrix.add        (local_dof_indices[i],local_dof_indices[j],cell_mass_matrix(i,j)       );
  	      }
  	    system_rhs(local_dof_indices[i]) += cell_rhs(i);
  	  }
  	assembler_lock.release ();
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::
  assemble_system_pressure(const double previous_new_temperature,
			   const double previous_new_pressure)
  {  
    system_rhs_pressure.reinit         (dof_handler_pressure.n_dofs());
    system_matrix_pressure.reinit      (sparsity_pattern_pressure);
    mass_matrix_pressure.reinit        (sparsity_pattern_pressure);
    laplace_matrix_new_pressure.reinit (sparsity_pattern_pressure);
    laplace_matrix_old_pressure.reinit (sparsity_pattern_pressure);

    Vector<double> tmp(solution_pressure.size ());
    
    const unsigned int n_threads = multithread_info.n_threads();
    Threads::ThreadGroup<> threads;
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
    std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
      thread_ranges
      = Threads::split_range<active_cell_iterator> (dof_handler_pressure.begin_active(),
						    dof_handler_pressure.end(),
						    n_threads);
    for (unsigned int thread=0; thread<n_threads; ++thread)
      threads += Threads::new_thread (&Heat_Pipe::assemble_system_interval_pressure,
				      *this,
				      thread_ranges[thread].first,
				      thread_ranges[thread].second,
				      previous_new_temperature,
				      previous_new_pressure);
    threads.join_all ();
    
    //system_rhs_pressure.print(std::cout);

    mass_matrix_pressure.vmult        ( tmp,old_solution_pressure);
    system_rhs_pressure.add           ( 1.0,tmp);
    laplace_matrix_old_pressure.vmult ( tmp,old_solution_pressure);
    system_rhs_pressure.add           (-(1 - theta_pressure) * time_step,tmp);

    system_matrix_pressure.copy_from (mass_matrix_pressure);
    system_matrix_pressure.add       (theta_pressure * time_step, laplace_matrix_new_pressure);

    hanging_node_constraints_pressure.condense (system_matrix_pressure);
    hanging_node_constraints_pressure.condense (system_rhs_pressure);

    if ((wilting_point_reached==true) ||
	 (saturation_point_reached==true))
      {
	double surface_pressure=0.;
	if (saturation_point_reached==true)
	  surface_pressure=-10./*saturated_water_pressure*/;
	if (wilting_point_reached==true)
	  surface_pressure=wilting_water_pressure;
	
    	std::map<unsigned int,double> boundary_values;
    	VectorTools::interpolate_boundary_values (dof_handler_pressure,
    						  1,
    						  ConstantFunction<dim>(surface_pressure),
    						  boundary_values);
	
    	MatrixTools::apply_boundary_values (boundary_values,
    					    system_matrix_pressure,
    					    solution_pressure,
    					    system_rhs_pressure);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::assemble_system_interval_pressure(const typename DoFHandler<dim>::active_cell_iterator &begin,
							 const typename DoFHandler<dim>::active_cell_iterator &end,
							 const double previous_new_temperature,
							 const double previous_new_pressure)
  {
    QGauss<dim>       quadrature_formula(3);
    QGauss<dim-1>     face_quadrature_formula (3);
  
    FEValues<dim>     fe_values      (fe_pressure, quadrature_formula,
    				      update_values | update_gradients |
    				      update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe_pressure, face_quadrature_formula,
    				      update_values | update_gradients |
    				      update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell   = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size ();
  
    FullMatrix<double> cell_mass_matrix        (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_new (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_old (dofs_per_cell,dofs_per_cell);
    Vector<double>     cell_rhs                (dofs_per_cell);  
  
    std::vector<unsigned int> local_dof_indices (fe_pressure.dofs_per_cell);
    std::vector<double> old_pressure_values(n_q_points);
    std::vector<double> new_pressure_values(n_q_points);

    double face_boundary_indicator;
    unsigned int cell_number=0;
    typename DoFHandler<dim>::active_cell_iterator cell;
    for (cell=begin; cell!=end; ++cell)
      {
	fe_values.reinit (cell);
	cell_mass_matrix        = 0;
	cell_laplace_matrix_new = 0;
	cell_laplace_matrix_old = 0;
	cell_rhs                = 0;
	
	fe_values.get_function_values(old_solution_pressure,
				      old_pressure_values);
	fe_values.get_function_values(solution_pressure,
				      new_pressure_values);
	cell_number++;
	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	  {
	    double average_pressure
	      =(0.5*old_pressure_values[q_point]+
		0.5*new_pressure_values[q_point]);
	    
	    double specific_moisture_capacity
	      =(-1./hydraulic_constant_b)*(saturated_moisture_content/saturated_water_pressure)
	      *pow(average_pressure
		   /saturated_water_pressure,(-1.-hydraulic_constant_b)/hydraulic_constant_b);
	    double unsaturated_hydraulic_conductivity
	      =saturated_hydraulic_conductivity
	      *pow(average_pressure
		   /saturated_water_pressure,-1.*(2.*hydraulic_constant_b+3.)/hydraulic_constant_b);

	    double moisture_content
	      =saturated_moisture_content
	      *pow(average_pressure
		   /saturated_water_pressure,(-1./hydraulic_constant_b));
	    
	    // if ((cell_number==1) &&
	    // 	(q_point==0)&&
	    // 	(timestep_number>=107))
	    //   {
	    // 	std::cout << "old_pressure_value: " << old_pressure_values[q_point] << std::endl;
	    // 	std::cout << "new_pressure_value: " << new_pressure_values[q_point] << std::endl;
	    // 	std::cout << "moisture_content  : " << moisture_content << std::endl;
	    // 	std::cout << "specific_moisture_capacity: " << std::scientific << specific_moisture_capacity << std::endl;
	    // 	std::cout << "unsaturated_hydraulic_conductivity: " << std::scientific << unsaturated_hydraulic_conductivity << std::endl;
	    //   }
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
 		    cell_mass_matrix(i,j)+=(fe_values.shape_value (i,q_point) *
					    specific_moisture_capacity *
					    fe_values.shape_value (j,q_point) *
					    fe_values.JxW (q_point));
		    cell_laplace_matrix_new(i,j)+=(unsaturated_hydraulic_conductivity *
						   fe_values.shape_grad (i, q_point) *
						   fe_values.shape_grad (j, q_point) *
						   fe_values.JxW (q_point));
		    cell_laplace_matrix_old(i,j)+=(unsaturated_hydraulic_conductivity *
						   fe_values.shape_grad (i, q_point) *
						   fe_values.shape_grad (j, q_point) *
						   fe_values.JxW (q_point));
		  }
		cell_rhs(i)+=-1.*(time_step *
				  unsaturated_hydraulic_conductivity *
				  fe_values.shape_grad (i,q_point)[0] *
				  fe_values.JxW (q_point));
	      }
	  }
	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	  {
	    face_boundary_indicator = cell->face(face)->boundary_indicator();
	    if ((cell->face(face)->at_boundary()) &&
		(face_boundary_indicator==1) 
		&&
		((old_precipitation<0.01) &&
		 (new_precipitation<0.01)))
	      {
		fe_face_values.reinit (cell,face);

		double old_surface_temperature
		  =VectorTools::point_value(dof_handler,
					    old_solution,
					    Point<dim>(0.));
		double new_surface_temperature
		  =previous_new_temperature;

		double old_surface_pressure 
		  =VectorTools::point_value(dof_handler_pressure,
					    old_solution_pressure,
					    Point<dim>(0.));
		double new_surface_pressure 
		  =previous_new_pressure;

		BoundaryConditions boundary_condition_old (analytic,
							   time_step*(timestep_number-1) + initial_time,
							   old_air_temperature,
							   old_solar_radiation,
							   old_wind_speed,
							   old_relative_humidity,
							   old_precipitation,
							   old_surface_temperature,
							   old_surface_pressure);
		BoundaryConditions boundary_condition_new (analytic,
							   time_step*timestep_number + initial_time,
							   new_air_temperature,
							   new_solar_radiation,
							   new_wind_speed,
							   new_relative_humidity,
							   new_precipitation,
							   new_surface_temperature,
							   new_surface_pressure);
		double new_evaporative_mass_flux
		  = boundary_condition_new
		  .get_evaporative_mass_flux(surface_type,
					     author,
					     canopy_density,
					     old_surface_temperature,
					     new_surface_temperature,
					     true);
		double old_evaporative_mass_flux
		  = boundary_condition_old
		  .get_evaporative_mass_flux(surface_type,
					     author,
					     canopy_density,
					     old_surface_temperature,
					     new_surface_temperature,
					     false);

		soil_heat_fluxes[timestep_number-1][10]=new_evaporative_mass_flux;
		
		if ((old_precipitation<0.01)&&
		    (new_precipitation<0.01)&&
		    (new_evaporative_mass_flux<0.))
		  saturation_point_reached=false;
		if (new_evaporative_mass_flux>0.)
		  wilting_point_reached=false;

		if ((wilting_point_reached==false) &&
		    (saturation_point_reached==false))
		  {
		    for (unsigned int q_face_point=0; q_face_point<n_face_q_points; ++q_face_point)
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			cell_rhs(i)+=((time_step * theta_pressure * (1./1000.) *
				       new_evaporative_mass_flux *
				       fe_face_values.shape_value (i,q_face_point) *
				       fe_face_values.JxW (q_face_point))
				      +
				      (time_step * (1-theta_pressure) * (1./1000.) *
				       old_evaporative_mass_flux *
				       fe_face_values.shape_value (i,q_face_point) *
				       fe_face_values.JxW (q_face_point)));
		  }
	      }
	  }
	
	cell->get_dof_indices (local_dof_indices);
	
	assembler_lock.acquire ();
	
	for (unsigned int i=0; i<dofs_per_cell; ++i)
	  {
	    for (unsigned int j=0; j<dofs_per_cell; ++j)
	      {
		laplace_matrix_new_pressure.add (local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_new(i,j));
		laplace_matrix_old_pressure.add (local_dof_indices[i],local_dof_indices[j],cell_laplace_matrix_old(i,j));
		mass_matrix_pressure.add        (local_dof_indices[i],local_dof_indices[j],cell_mass_matrix(i,j)       );
	      }
	    system_rhs_pressure(local_dof_indices[i]) += cell_rhs(i);
	  }
	assembler_lock.release ();
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::solve_temperature()
  {
    SolverControl solver_control (solution.size(),
  				  1e-8*system_rhs.l2_norm ());
    SolverCG<> cg   (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize (system_matrix, 1.2);
    
    cg.solve (system_matrix, solution, system_rhs,
  	      preconditioner);
    
    hanging_node_constraints.distribute (solution);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::solve_pressure ()
  {
    SolverControl solver_control(solution_pressure.size(),
				 1e-8*system_rhs_pressure.l2_norm ());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize (system_matrix_pressure, 1.2);
    
    cg.solve(system_matrix_pressure,solution_pressure,
	     system_rhs_pressure,preconditioner);
    
    hanging_node_constraints_pressure.distribute(solution_pressure);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::output_results () const
  {
    DataOut<dim> data_out;
    
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector    (solution,"solution");
    data_out.build_patches ();
    
    std::stringstream t;
    t << timestep_number;
    
    std::stringstream d;
    d << dim;
    
    std::string filename = "./output/solution_" 
      + d.str() + "d_time_" 
      + t.str() + ".gp";
    
    std::ofstream output (filename.c_str());
    data_out.write_gnuplot (output);      
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::fill_output_vectors()
  {
    Names names(preheating_step,
		met_data_type);
    unsigned int soil_depths = names.road_depths.size();
    std::vector<double> soil_bha_temperature_row(soil_depths,0.);
    std::vector<double> soil_bha_thermal_conductivity_row(soil_depths,0.);
    std::vector<double> soil_bha_thermal_heat_capacity_row(soil_depths,0.);
    std::vector<double> soil_bha_density_row(soil_depths,0.);

    std::vector<double> soil_bha_pressure_row(soil_depths,0.);
    std::vector<double> soil_bha_moisture_content_row(soil_depths,0.);
    std::vector<double> soil_bha_hydraulic_conductivity_row(soil_depths,0.);
    std::vector<double> soil_bha_hydraulic_moisture_capacity_row(soil_depths,0.);
	  
    for (unsigned int i=0; i<soil_depths; i++)
      {
	double moisture_content=0.23;
	double specific_moisture_capacity=0.;
	double unsaturated_hydraulic_conductivity=0.;

	soil_bha_temperature_row[i]
	  =VectorTools::point_value(dof_handler,old_solution,
				    borehole_A_depths[i]);
	if (moisture_movement==true)
	  {
	    soil_bha_pressure_row[i]
	      =VectorTools::point_value(dof_handler_pressure,old_solution_pressure,
					borehole_A_depths[i]);
	    moisture_content
	      =saturated_moisture_content
	      *pow(soil_bha_pressure_row[i]
		   /saturated_water_pressure,(-1./hydraulic_constant_b));
	    specific_moisture_capacity
	      =(-1./hydraulic_constant_b)*(saturated_moisture_content/saturated_water_pressure)
	      *pow(soil_bha_pressure_row[i]
		   /saturated_water_pressure,(-1.-hydraulic_constant_b)/hydraulic_constant_b);
	    unsaturated_hydraulic_conductivity
	      =saturated_hydraulic_conductivity
	      *pow(soil_bha_pressure_row[i]
		   /saturated_water_pressure,-1.*(2.*hydraulic_constant_b+3.)/hydraulic_constant_b);

	    soil_bha_moisture_content_row[i]
	      =moisture_content;
	    soil_bha_hydraulic_conductivity_row[i]
	      =unsaturated_hydraulic_conductivity;
	    soil_bha_hydraulic_moisture_capacity_row[i]
	      =specific_moisture_capacity;
	  }
	
	MaterialData material_data (dim,false,moisture_content,moisture_movement);
	double thermal_conductivity  = material_data.get_soil_thermal_conductivity(3);
	double thermal_heat_capacity = material_data.get_soil_heat_capacity(3);
	double density               = material_data.get_soil_density(3);	      
	soil_bha_thermal_conductivity_row[i]
	  =thermal_conductivity;
	soil_bha_thermal_heat_capacity_row[i]
	  =thermal_heat_capacity;
	soil_bha_density_row[i]
	  =density;
      }
    soil_bha_temperature.push_back(soil_bha_temperature_row);
    soil_bha_thermal_conductivity.push_back(soil_bha_thermal_conductivity_row);
    soil_bha_thermal_heat_capacity.push_back(soil_bha_thermal_heat_capacity_row);
    soil_bha_density.push_back(soil_bha_density_row);
    if (moisture_movement==true)
      {
	soil_bha_pressure.push_back(soil_bha_pressure_row);
	soil_bha_moisture_content.push_back(soil_bha_moisture_content_row);
	soil_bha_hydraulic_conductivity.push_back(soil_bha_hydraulic_conductivity_row);
	soil_bha_hydraulic_moisture_capacity.push_back(soil_bha_hydraulic_moisture_capacity_row);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::update_met_data ()
  {
    if (analytic)
      {
  	AnalyticSolution analytic_solution(0,0,0,"");
  	old_solar_radiation   = analytic_solution.get_analytic_solar_radiation   (initial_time);
  	old_air_temperature   = analytic_solution.get_analytic_air_temperature   (initial_time);
  	old_relative_humidity = analytic_solution.get_analytic_relative_humidity (/*initial_time*/);
  	old_wind_speed        = analytic_solution.get_analytic_wind_speed        (/*initial_time*/);
  	old_precipitation     = analytic_solution.get_analytic_precipitation     (/*initial_time*/);
      }
    else
      {
  	if (date_and_time.size()==0 &&
  	    met_data.size()     ==0)
  	  {
  	    read_met_data (date_and_time,
  			   met_data,
  			   time_step,
  			   preheating_step,
  			   met_data_type);
  	    std::cout << "\tAvailable met lines: " << met_data.size()
  		      << std::endl << std::endl;
  	  }
  	old_air_temperature   = met_data[timestep_number-1][0];
  	old_relative_humidity = met_data[timestep_number-1][1];
  	old_wind_speed        = met_data[timestep_number-1][2];
  	old_wind_direction    = met_data[timestep_number-1][3];
  	old_solar_radiation   = met_data[timestep_number-1][4];
  	old_precipitation     = met_data[timestep_number-1][5];
       
  	new_air_temperature   = met_data[timestep_number][0];
  	new_relative_humidity = met_data[timestep_number][1];
  	new_wind_speed        = met_data[timestep_number][2];
  	new_wind_direction    = met_data[timestep_number][3];
  	new_solar_radiation   = met_data[timestep_number][4];
  	new_precipitation     = met_data[timestep_number][5];
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::initial_condition_temperature()
  {
    VectorTools::project (dof_handler,
			  hanging_node_constraints,
			  QGauss<dim>(3),
			  InitialValue<dim>(10.),
			  old_solution);
    solution=old_solution;
    // VectorTools::project (dof_handler,
    // 			  hanging_node_constraints,
    // 			  QGauss<dim> (3),
    // 			  ConstantFunction<dim> (10),
    // 			  InitialValue<dim>(surface_type,0),
    // 			  solution);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template <int dim>
  void Heat_Pipe<dim>::run()
  {
    read_grid_temperature();
    setup_system_temperature();
    solution.reinit (dof_handler.n_dofs());
    old_solution.reinit (dof_handler.n_dofs());
    initial_condition_temperature();
    
    if(moisture_movement==true)
      {
	std::cout << "Solving heat equation with moisture movement\n";
	read_grid_pressure();
	setup_system_pressure();
	solution_pressure.reinit(dof_handler_pressure.n_dofs());
	old_solution_pressure.reinit(dof_handler_pressure.n_dofs());
	VectorTools::project(dof_handler_pressure,
			     hanging_node_constraints_pressure,
			     QGauss<dim> (3),
			     ConstantFunction<dim> (-10./*saturated_water_pressure*//*-75.2025*/),
			     old_solution_pressure);
	VectorTools::project(dof_handler_pressure,
			     hanging_node_constraints_pressure,
			     QGauss<dim> (3),
			     ConstantFunction<dim> (-10./*saturated_water_pressure*//*-75.2025*/),
			     solution_pressure);
      }
    else
      std::cout << "Solving heat equation without moisture movement\n";
    
    const unsigned int n_threads = multithread_info.n_threads();
    std::cout << "   Number of threads: " << n_threads << std::endl;
    std::cout << "   Number of active cells for temperature:       "
	      << triangulation.n_active_cells()
	      << std::endl;
    std::cout << "   Number of degrees of freedom for temperature: "
	      << dof_handler.n_dofs()
	      << std::endl;
    if (moisture_movement==true)
      {
	std::cout << "   Number of active cells for pressure:       "
		  << triangulation_pressure.n_active_cells()
		  << std::endl;
	std::cout << "   Number of degrees of freedom for pressure: "
		  << dof_handler_pressure.n_dofs()
		  << std::endl;
      }

    int ref_day=0;
    int ref_month=0;
    int ref_hour=0;
    int ref_min=0;
    int ref_sec=0;
    double previous_surface_temperature=10.;
    double current_surface_temperature=10.;
    double previous_surface_pressure=-10./*saturated_water_pressure*/;
    double current_surface_pressure=-10./*saturated_water_pressure*/;
    for (time=time_step, timestep_number=1;
	 time<=time_max;
	 time+=time_step, ++timestep_number)
      {
	update_met_data();
	canopy_density=canopy_density_period_1;
	if (author=="Best")
	  for (unsigned int i=0; i<20; i++)
	    if ((timestep_number>=(2184+8760*i)*(3600./time_step)) && 
		(timestep_number<=(5088+8760*i)*(3600./time_step)))
	      {
		canopy_density=canopy_density_period_2;
		break;
	      }
	fill_output_vectors();
	std::cout << "Time step " << timestep_number << "\t"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][0] << "/"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][1] << "/"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][2] << "\t"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][3] << ":"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][4] << ":"
		  << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][5];
	std::cout.setf( std::ios::fixed, std::ios::floatfield );
	std::cout << "\tTa: " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_air_temperature
		  << "\tR : " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_solar_radiation
		  << "\tUs: " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_wind_speed
		  << "\tHr: " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_relative_humidity
		  << "\tI : " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_precipitation
		  << "\tc " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << canopy_density
		  << std::endl;
	if (timestep_number==1)
	  {
	    ref_day  =date_and_time[timestep_number][0];
	    ref_month=date_and_time[timestep_number][1];
	    ref_hour =date_and_time[timestep_number][3];
	    ref_min  =date_and_time[timestep_number][4];
	    ref_sec  =date_and_time[timestep_number][5];
	  }
	unsigned int step=0;
	double tolerance_temperature=1000.;
	double tolerance_pressure=1000.;
       	std::vector <double> temp (11,0.);
	soil_heat_fluxes.push_back(temp);
	do
	  {
	    if(moisture_movement==true)
	      {
		/*---solve pressure---*/
		assemble_system_pressure(current_surface_temperature,
					 current_surface_pressure);
		solve_pressure();
		/*--------------------*/
		double new_surface_pressure
		  =VectorTools::point_value(dof_handler_pressure,
					    solution_pressure,
					    Point<dim>(0.));
		wilting_point_reached=false;
		saturation_point_reached=false;
		if (new_surface_pressure<=wilting_water_pressure)
		  {
		    wilting_point_reached=true;
		    current_surface_pressure=wilting_water_pressure;
		  }
		else if ((old_precipitation>=0.01)||
			 (new_precipitation>=0.01)||
			 (new_surface_pressure>=-10.))
		  {
		    saturation_point_reached=true;
		    current_surface_pressure=-10.;
		  }
		else
		  current_surface_pressure
		    =0.5*previous_surface_pressure
		    +0.5*new_surface_pressure;
	      }
	    /*---solve temperature---*/
	    assemble_system_temperature(current_surface_temperature,
					current_surface_pressure);
	    solve_temperature();
	    /*--------------------*/
	    current_surface_temperature
	      = 0.5*previous_surface_temperature
	      + 0.5*VectorTools::point_value(dof_handler,
					     solution,
					     Point<dim>(0.));
	    tolerance_temperature
	      =fabs(previous_surface_temperature -
		    current_surface_temperature);
	    tolerance_pressure
	      =fabs(previous_surface_pressure-
		    current_surface_pressure);
	    // if (timestep_number>=107)
	    //   {
	    // 	std::cout.setf( std::ios::fixed, std::ios::floatfield );
	    // 	std::cout << std::fixed << "\t" << "step: " << step << "\n"
	    // 		  << "\told Ts: " << std::setfill(' ') << std::setprecision(2) << previous_surface_temperature
	    // 		  << "\tnew Ts: " << std::setfill(' ') << std::setprecision(2) << current_surface_temperature
	    // 		  << "\tdT: " << std::setfill(' ') << std::setprecision(2) << tolerance_temperature << std::endl
	    // 		  << "\told Ps: " << std::setfill(' ') << std::setprecision(2) << previous_surface_pressure
	    // 		  << "\tnew Ps: " << std::setfill(' ') << std::setprecision(2) << current_surface_pressure
	    // 		  << "\tdP: " << std::setfill(' ') << std::setprecision(2) << tolerance_pressure << std::endl
	    // 		  << "\twilting_point_reached: " << wilting_point_reached << std::endl
	    // 		  << "\tsaturation_point_reached: " << saturation_point_reached << std::endl;
	    // 	for (unsigned int i=0; i<soil_heat_fluxes[timestep_number-1].size(); i++)
	    // 	  std::cout << "\t" << soil_heat_fluxes[timestep_number-1][i];
	    // 	std::cout << std::endl;
	    //   }
	    step++;
	    
	    previous_surface_temperature=current_surface_temperature;
	    previous_surface_pressure   =current_surface_pressure;
	  }
	while ((tolerance_temperature>0.3) ||
	       (tolerance_pressure>0.1));
	// output results
	// if (ref_day   == date_and_time[timestep_number][0] &&
	//     ref_month == date_and_time[timestep_number][1] &&
	//     ref_hour  == date_and_time[timestep_number][3] &&
	//     ref_min   == date_and_time[timestep_number][4] &&
	//     ref_sec   == date_and_time[timestep_number][5])
	//   output_results ();
	
	if (timestep_number==timestep_number_max) 
	  {
	    std::vector< std::vector<int> >::const_iterator 
	      first=date_and_time.begin(), second=date_and_time.begin()+timestep_number_max;
	    std::vector< std::vector<int> > date_and_time_1d(first,second);
	    
	    std::stringstream cd_1;
	    cd_1 << std::fixed << std::setprecision(2) << canopy_density_period_1;
	    std::stringstream cd_2;
	    cd_2 << std::fixed << std::setprecision(2) << canopy_density_period_2;
	    std::string canopy_densities
	      =cd_1.str()+"_"+cd_2.str();

	    std::vector< std::string > filenames;
	    filenames.push_back(filename_preffix + "_bha_heat_fluxes_" + canopy_densities + ".txt");
	    filenames.push_back(filename_preffix + "_bha_temperatures_" + canopy_densities + ".txt");
	    filenames.push_back(filename_preffix + "_bha_thermal_conductivity_" + canopy_densities + ".txt");
	    filenames.push_back(filename_preffix + "_bha_thermal_heat_capacity_" + canopy_densities + ".txt");
	    filenames.push_back(filename_preffix + "_bha_density_" + canopy_densities + ".txt");
	    if (moisture_movement==true)
	      {
		filenames.push_back(filename_preffix + "_bha_pressures_" + canopy_densities + ".txt");
		filenames.push_back(filename_preffix + "_bha_moisture_content_" + canopy_densities + ".txt");
		filenames.push_back(filename_preffix + "_bha_hydraulic_conductivity_" + canopy_densities + ".txt");
		filenames.push_back(filename_preffix + "_bha_hydraulic_moisture_capacity_" + canopy_densities + ".txt");
	      }
	    
	    std::vector< std::vector < std::vector<double> > > output_data;
	    output_data.push_back(soil_heat_fluxes);
	    output_data.push_back(soil_bha_temperature);
	    output_data.push_back(soil_bha_thermal_conductivity);
	    output_data.push_back(soil_bha_thermal_heat_capacity);
	    output_data.push_back(soil_bha_density);
	    if (moisture_movement==true)
	      {
		output_data.push_back(soil_bha_pressure);
		output_data.push_back(soil_bha_moisture_content);
		output_data.push_back(soil_bha_hydraulic_conductivity);
		output_data.push_back(soil_bha_hydraulic_moisture_capacity);
	      }
	    
	    for (unsigned int i=0; i<filenames.size(); i++)
	      {
		//std::string filename="soil_heat_fluxes.txt";
		std::ofstream file (filenames[i].c_str());
		if (!file.is_open())
		  throw 2;
		std::cout << "Writing file: " << filenames[i] << std::endl;
		print_data (0,
			    date_and_time_1d,
			    output_data[i],
			    file);
		file.close();
		if (file.is_open())
		  throw 3;
	      }
	  }
	old_solution = solution;
	old_solution_pressure = solution_pressure;
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
