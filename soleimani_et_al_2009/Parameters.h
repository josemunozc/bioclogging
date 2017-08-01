namespace Parameters
{
  using namespace dealii;
  template <int dim>
  struct AllParameters
  {
    AllParameters ();
    static void declare_parameters (ParameterHandler &prm);
    void parse_parameters (ParameterHandler &prm);

    unsigned int timestep_number_max;
    double time_step;
    double theta_richards;
    double theta_transport;
    double domain_size;
    double insulation_thickness;
    double insulation_depth;
    unsigned int refinement_level;
    unsigned int output_frequency_transport;

    double soil_thermal_conductivity;
    double soil_density;
    double soil_specific_heat_capacity;
    double insulation_thermal_conductivity;
    double insulation_density;
    double insulation_specific_heat_capacity;

    double initial_condition_homogeneous_flow;
    double initial_condition_homogeneous_transport;
    double initial_condition_homogeneous_bacteria;
    bool initial_condition_using_files;
    std::string file_with_initial_condition_flow;
    std::string file_with_initial_condition_transport;
    std::string file_with_initial_condition_bacteria;

    bool richards_fixed_at_bottom;
    double richards_bottom_fixed_value;
    double richards_bottom_flow_value;

    bool richards_fixed_at_top;
    double richards_top_fixed_value;
    double richards_top_flow_value;

    bool transport_fixed_at_top;
    double transport_top_fixed_value;

    double van_genuchten_n;
    double van_genuchten_alpha;

    double first_order_decay_factor;
    double yield_coefficient;
    double maximum_substrate_use_rate;
    double half_velocity_constant;
    double decay_rate;
    double dry_density;

    bool test_function_transport;
    bool use_mesh_file;
    bool lumped_matrix;
    bool homogeneous_decay_rate;
    bool coupled_transport;
    double biomass_dry_density;

    std::string mesh_filename;
    std::string moisture_transport_equation;
    std::string hydraulic_properties;
    std::string output_file_format;
    std::string initial_state;
    std::string transport_mass_entry_point;
    std::string output_directory;
    bool output_data_in_terminal;

    // Hydraulic properties
    double saturated_hydraulic_conductivity;
    double moisture_content_saturation;
    double moisture_content_residual;
    double porosity;
    double dispersivity_longitudinal;
    double dispersivity_transverse;
    double effective_diffusion_coefficient;
    std::string relative_permeability_model;
    std::string sand_fraction;
};

  template <int dim>
  AllParameters<dim>::AllParameters ()
  {
    timestep_number_max       =0;
    time_step                 =0.;
    theta_richards            =0.;
    theta_transport           =0.;
    domain_size               =0.;
    insulation_thickness      =0.;
    insulation_depth          =0.;
    refinement_level          =0.;
    output_frequency_transport=0;

    soil_thermal_conductivity        =0.;
    soil_density                     =0.;
    soil_specific_heat_capacity      =0.;
    insulation_thermal_conductivity  =0.;
    insulation_density               =0.;
    insulation_specific_heat_capacity=0.;

    initial_condition_homogeneous_flow=0.;
    initial_condition_homogeneous_transport=0.;
    initial_condition_homogeneous_bacteria=0.;
    initial_condition_using_files=false;

    richards_fixed_at_bottom=false;
    richards_bottom_fixed_value=0.;
    richards_bottom_flow_value=0.;

    richards_fixed_at_top=false;
    richards_top_fixed_value=0.;
    richards_top_flow_value=0.;

    transport_fixed_at_top=false;
    transport_top_fixed_value=0.;

    van_genuchten_n    =0.;
    van_genuchten_alpha=0.;

    first_order_decay_factor  =0.;
    yield_coefficient         =0.;
    maximum_substrate_use_rate=0.;
    half_velocity_constant    =0.;
    decay_rate                =0.;
    dry_density               =0.;

    test_function_transport=false;
    use_mesh_file          =false;
    lumped_matrix          =false;
    homogeneous_decay_rate =false;
    coupled_transport      =false;
    biomass_dry_density    =0.;

    saturated_hydraulic_conductivity=0.;
    moisture_content_saturation     =0.;
    moisture_content_residual       =0.;
    porosity                        =0.;
    dispersivity_longitudinal       =0.;
    dispersivity_transverse         =0.;
    effective_diffusion_coefficient =0.;

    output_data_in_terminal=false;
}

  template <int dim>
  void
  AllParameters<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection("time stepping");
    {
      prm.declare_entry("time step", "3600",
			Patterns::Double(0),
			"simulation time step");
      prm.declare_entry("timestep number max", "70055",
			Patterns::Integer(),
			"max number of timesteps to execute");
      prm.declare_entry("richards theta scheme value", "0.5",
			Patterns::Double(0,1),
			"value for theta that interpolated between explicit "
			"Euler (theta=0), Crank-Nicolson (theta=0.5), and "
			"implicit Euler (theta=1).");
      prm.declare_entry("transport theta scheme value", "0.5",
			Patterns::Double(0,1),
			"value for theta that interpolated between explicit "
			"Euler (theta=0), Crank-Nicolson (theta=0.5), and "
			"implicit Euler (theta=1).");
    }
    prm.leave_subsection();

    prm.enter_subsection("geometric data");
    {
      prm.declare_entry("use mesh file", "true",
			Patterns::Bool(),
			"if true, use the mesh file specified by mesh_filename");
      prm.declare_entry("mesh filename", "soleimani_mesh.msh",
			Patterns::Anything(),
			"this variable specifies the location of the mesh file");
      prm.declare_entry("domain size", "20",
			Patterns::Double(0),
			"size of domain in m");
      prm.declare_entry("insulation thickness", "0.20",
			Patterns::Double(0),
			"thickness of insulation in m");
      prm.declare_entry("insulation depth", "1.00",
			Patterns::Double(0),
			"depth of insulation layer in m");
      prm.declare_entry("refinement level", "5",
			Patterns::Integer(),
			"number of cells as in 2^n");
    }
    prm.leave_subsection();

    prm.enter_subsection("material data");
    {
      prm.declare_entry("soil thermal conductivity", "1.2",
			Patterns::Double(0),
			"thermal conductivity of soil in W/mK");
      prm.declare_entry("soil density", "1960.",
			Patterns::Double(0),
			"density of soil in kg/m3");
      prm.declare_entry("soil specific heat capacity", "840.",
			Patterns::Double(0),
			"specific capacity of soil in J/mK");
      prm.declare_entry("insulation thermal conductivity", "0.034",
			Patterns::Double(0),
			"thermal conductivity of insulation in W/mK");
      prm.declare_entry("insulation density", "30.",
			Patterns::Double(0),
			"density of insulation in kg/m3");
      prm.declare_entry("insulation specific heat capacity", "1130.",
			Patterns::Double(0),
			"specific capacity of insulation in J/mK");
    }
    prm.leave_subsection();

    prm.enter_subsection("initial conditions");
    {
      prm.declare_entry("initial state", "no_drying",
			Patterns::Anything(), "a default value will initialise the soil "
			"with default values, other options will load corresponding "
			"files to fill solution vectors");
      prm.declare_entry("initial condition homogeneous flow", "0.0",
			Patterns::Double(), "homogeneous value for flow "
			"equation.");
      prm.declare_entry("initial condition homogeneous transport", "0.0",
			Patterns::Double(), "homogeneous value for transport "
			"equation.");
      prm.declare_entry("initial condition homogeneous bacteria", "0.0",
			Patterns::Double(), "homogeneous value for initial bacteria "
			"concentration in the domain.");
      prm.declare_entry("initial condition using files", "false",
			Patterns::Bool(), "If true use specified files to load "
			"initial conditions for flow and transport equations and "
			"bacteria concentration.");
      prm.declare_entry("file with initial condition flow", "final_state_flow.ph",
			Patterns::Anything(), "path to file with initial conditions for "
			"flow equation.");
      prm.declare_entry("file with initial condition transport", "final_state_transport.ph",
			Patterns::Anything(), "path to file with initial conditions for "
			"transport equation.");
      prm.declare_entry("file with initial condition bacteria", "final_state_bacteria.ph",
			Patterns::Anything(), "path to file with initial conditions for "
			"bacteria concentration.");
    }
    prm.leave_subsection();

    prm.enter_subsection("boundary conditions");
    {
      prm.declare_entry("richards fixed at bottom", "false",
			Patterns::Bool(),
			"if true, the bottom boundary condition is fixed");
      prm.declare_entry("richards bottom fixed value", "10.",
			Patterns::Double(),
			"value at bottom boundary for fixed conditions");
      prm.declare_entry("richards bottom flow value", "0.001",
			Patterns::Double(),
			"value at bottom boundary for flow conditions");

      prm.declare_entry("richards fixed at top", "true",
			Patterns::Bool(),
			"if true, the top boundary condition is fixed");
      prm.declare_entry("richards top fixed value","0.",
			Patterns::Double(),
			"value at top boundary for fixed conditions");
      prm.declare_entry("richards top flow value","0.",
			Patterns::Double(),
			"value at top boundary for flow conditions");
      
      prm.declare_entry("transport fixed at top", "false",
			Patterns::Bool(),
			"if true, the top boundary condition is fixed");
      prm.declare_entry("transport top fixed value", "0.",
			Patterns::Double(),
			"value at top boundary for fixed conditions");
      prm.declare_entry("transport mass entry point", "top",
			Patterns::Anything(), "define the entry point "
			"for mass in the transpor equation");

    }
    prm.leave_subsection();


    prm.enter_subsection("equations");
    {
      prm.declare_entry("moisture transport", "mixed",
			Patterns::Anything(), "define the type of "
			"equation used to solve moisture "
			"movement in the domain");
      prm.declare_entry("hydraulic properties", "van_genuchten_1980",
			Patterns::Anything(), "define the set of "
			"equations used to define the hydraulic "
			"properties of the domain");
      prm.declare_entry("lumped matrix (qtrapez)", "false",
			Patterns::Bool(), "use the trapezoidal rule for "
			"numerical integration. This produces a lumped "
			"matrix and avoid oscillations when solving "
			"certain kind of problems (e.g. richards' equation."
			"see Celia et al 1990 and this link for more "
			"information https://www.dealii.org/archive/dealii/msg03673.html");

      prm.declare_entry("homogeneous decay rate", "false",
			Patterns::Bool(), "activate an homogeneous "
			"1st order decay rate factor in transport "
			"equation ");
      prm.declare_entry("coupled transport", "false",
			Patterns::Bool(), "add the transport equation to the system of "
			"equations to be solved.");
    }
    prm.leave_subsection();

    prm.enter_subsection("van genuchten parameters");
    {
      prm.declare_entry("van genuchten n","2",
			Patterns::Double(),"Define Van Genuchten"
			" n parameter.");
      prm.declare_entry("van genuchten alpha","1.611E6",
			Patterns::Double(),"Define Van Genuchten"
			" n parameter.");
    }
    prm.leave_subsection();

    prm.enter_subsection("hydraulic properties");
    {
      prm.declare_entry("sand fraction", "0um",
			Patterns::Anything(),"string defining the sand "
			"fraction used.");
      prm.declare_entry("saturated hydraulic conductivity","0.00922",
			Patterns::Double(),"Declare saturated hydraulic "
			"conductivity in cm/s");
      prm.declare_entry("moisture content saturation","0.368",
			Patterns::Double()," ");
      prm.declare_entry("moisture content residual","0.075",
			Patterns::Double()," ");
      prm.declare_entry("porosity","0.8",
			Patterns::Double()," ");
      prm.declare_entry("dispersivity longitudinal","1",
			Patterns::Double()," ");
      prm.declare_entry("dispersivity transverse","0",
			Patterns::Double()," ");
      prm.declare_entry("effective diffusion coefficient","0.00001",
			Patterns::Double()," ");
      prm.declare_entry("relative permeability model", "soleimani",
			Patterns::Anything(),"choose the model to calculate the "
			"relative hydraulic conductivity.");
    }
    prm.leave_subsection();

    prm.enter_subsection("reaction properties");
    {
      prm.declare_entry("first order decay factor","0.",Patterns::Double(),"Declare first order "
			"decay factor in 1/s");
      prm.declare_entry("yield coefficient","0.",Patterns::Double()," ");
      prm.declare_entry("maximum substrate use rate","0.",Patterns::Double()," ");
      prm.declare_entry("half velocity constant","0.",Patterns::Double()," ");
      prm.declare_entry("decay rate","0.",Patterns::Double()," ");
      prm.declare_entry("biomass dry density","0.",Patterns::Double()," ");
    }
    prm.leave_subsection();

    prm.enter_subsection("test functions");
    {
      prm.declare_entry("test function transport","false",
			Patterns::Bool(),"If true, the program will"
			" run a test of this function. Previous input"
			" parameters might be ignored.");
    }
    prm.leave_subsection();

    prm.enter_subsection("general things");
    {
      prm.declare_entry("output file format",".gp",
			Patterns::Anything(),"Define the output "
			"file format. Currently implemented .gp "
			"and .vtu");
      prm.declare_entry("output frequency transport", "60",
			Patterns::Integer(),"number that defines the"
			"number of seconds between outputs");
      prm.declare_entry("output directory", "output",
			Patterns::Anything(),"name of output "
			"directory to store all output files.");
      prm.declare_entry("output data in terminal", "true",
			Patterns::Bool(),"if true, the program will generate output "
			"in the terminal. Set to false to avoid cluttering "
			"and speed up a bit the program.");
    }
    prm.leave_subsection();
  }

  template <int dim>
  void AllParameters<dim>::parse_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection("time stepping");
    {
      time_step          =prm.get_double ("time step");
      timestep_number_max=prm.get_integer("timestep number max");
      theta_richards     =prm.get_double ("richards theta scheme value");
      theta_transport    =prm.get_double ("transport theta scheme value");
    }
    prm.leave_subsection();


    prm.enter_subsection("geometric data");
    {
      use_mesh_file       =prm.get_bool   ("use mesh file");
      mesh_filename       =prm.get        ("mesh filename");
      domain_size         =prm.get_double ("domain size");
      insulation_thickness=prm.get_double ("insulation thickness");
      insulation_depth    =prm.get_double ("insulation depth");
      refinement_level    =prm.get_integer("refinement level");
    }
    prm.leave_subsection();


    prm.enter_subsection("material data");
    {
      soil_thermal_conductivity        =prm.get_double("soil thermal conductivity");
      soil_density                     =prm.get_double("soil density");
      soil_specific_heat_capacity      =prm.get_double("soil specific heat capacity");
      insulation_thermal_conductivity  =prm.get_double("insulation thermal conductivity");
      insulation_density               =prm.get_double("insulation density");
      insulation_specific_heat_capacity=prm.get_double("insulation specific heat capacity");

    }
    prm.leave_subsection();

    prm.enter_subsection("initial conditions");
    {
      initial_state                          =prm.get("initial state");
      initial_condition_homogeneous_flow     =prm.get_double("initial condition homogeneous flow");
      initial_condition_homogeneous_transport=prm.get_double("initial condition homogeneous transport");
      initial_condition_homogeneous_bacteria =prm.get_double("initial condition homogeneous bacteria");
      initial_condition_using_files        =prm.get_bool("initial condition using files");
      file_with_initial_condition_flow     =prm.get("file with initial condition flow");
      file_with_initial_condition_transport=prm.get("file with initial condition transport");
      file_with_initial_condition_bacteria =prm.get("file with initial condition bacteria");
    }
    prm.leave_subsection();

    prm.enter_subsection("boundary conditions");
    {
      richards_fixed_at_bottom   =prm.get_bool  ("richards fixed at bottom");
      richards_bottom_fixed_value=prm.get_double("richards bottom fixed value");
      richards_bottom_flow_value =prm.get_double("richards bottom flow value");

      richards_fixed_at_top      =prm.get_bool  ("richards fixed at top");
      richards_top_fixed_value   =prm.get_double("richards top fixed value");
      richards_top_flow_value    =prm.get_double("richards top flow value");

      transport_fixed_at_top    =prm.get_bool  ("transport fixed at top");
      transport_top_fixed_value =prm.get_double("transport top fixed value");
      transport_mass_entry_point=prm.get       ("transport mass entry point");
    }
    prm.leave_subsection();

    prm.enter_subsection("equations");
    {
      moisture_transport_equation=prm.get("moisture transport");
      hydraulic_properties       =prm.get("hydraulic properties");
      lumped_matrix              =prm.get_bool("lumped matrix (qtrapez)");
      homogeneous_decay_rate     =prm.get_bool("homogeneous decay rate");
      coupled_transport          =prm.get_bool("coupled transport");
    }
    prm.leave_subsection();

    prm.enter_subsection("van genuchten parameters");
    {
      van_genuchten_n=prm.get_double("van genuchten n");
      van_genuchten_alpha=prm.get_double("van genuchten alpha");
    }
    prm.leave_subsection();

    prm.enter_subsection("hydraulic properties");
    {
      sand_fraction                   =prm.get       ("sand fraction");
      saturated_hydraulic_conductivity=prm.get_double("saturated hydraulic conductivity");
      moisture_content_saturation     =prm.get_double("moisture content saturation");
      moisture_content_residual       =prm.get_double("moisture content residual");
      porosity                        =prm.get_double("porosity");
      dispersivity_longitudinal       =prm.get_double("dispersivity longitudinal");
      dispersivity_transverse         =prm.get_double("dispersivity transverse");
      effective_diffusion_coefficient =prm.get_double("effective diffusion coefficient");
      relative_permeability_model     =prm.get       ("relative permeability model");
    }
    prm.leave_subsection();

    prm.enter_subsection("reaction properties");
    {
      first_order_decay_factor  =prm.get_double("first order decay factor");
      yield_coefficient         =prm.get_double("yield coefficient");
      maximum_substrate_use_rate=prm.get_double("maximum substrate use rate");
      half_velocity_constant    =prm.get_double("half velocity constant");
      decay_rate                =prm.get_double("decay rate");
      biomass_dry_density       =prm.get_double("biomass dry density");
    }
    prm.leave_subsection();

    prm.enter_subsection("test functions");
    {
      test_function_transport=prm.get_bool("test function transport");
    }
    prm.leave_subsection();

    prm.enter_subsection("general things");
    {
      output_file_format        =prm.get("output file format");
      output_frequency_transport=prm.get_integer("output frequency transport");
      output_directory	      =prm.get("output directory");
      output_data_in_terminal   =prm.get_bool("output data in terminal");
    }
    prm.leave_subsection();
  }
}
