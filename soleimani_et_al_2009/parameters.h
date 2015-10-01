
namespace Parameters
{  
  template <int dim>
    struct AllParameters
    {
      AllParameters ();
      
      unsigned int timestep_number_max;
      double time_step; 
      double theta;
      //std::string mesh_filename;
      double domain_size;
      double insulation_thickness;
      double insulation_depth;
      unsigned int refinement_level;

      double soil_thermal_conductivity;
      double soil_density;
      double soil_specific_heat_capacity;
      double insulation_thermal_conductivity;
      double insulation_density;
      double insulation_specific_heat_capacity;

      bool fixed_at_bottom;
      double bottom_fixed_value;
      bool fixed_at_top;
      std::string top_fixed_value_file;
      
      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };
    
  template <int dim>
    AllParameters<dim>::AllParameters ()
    {}
  
  template <int dim>
    void
    AllParameters<dim>::declare_parameters (ParameterHandler &prm)
    {
      // prm.declare_entry("mesh", "grid.inp",
      // 			Patterns::Anything(),
      // 			"intput file name");
      
      prm.enter_subsection("time stepping");
      {
	prm.declare_entry("time step", "3600",
			  Patterns::Double(0),
			  "simulation time step");
	prm.declare_entry("timestep number max", "70055",
			  Patterns::Integer(),
			  "max number of timesteps to execute");
	prm.declare_entry("theta scheme value", "0.5",
			  Patterns::Double(0,1),
			  "value for theta that interpolated between explicit "
			  "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
			  "implicit Euler (theta=1).");
      }
      prm.leave_subsection();


      prm.enter_subsection("geometric data");
      {
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


      prm.enter_subsection("boundary conditions");
      {
	prm.declare_entry("fixed at bottom", "false",
			  Patterns::Bool(),
			  "if true, the bottom boundary condtion is fixed");
	prm.declare_entry("bottom fixed value", "10.",
			  Patterns::Double(),
			  "value at bottom boundary for fixed conditions");
	prm.declare_entry("fixed at top", "true",
			  Patterns::Bool(),
			  "if true, the top boundary condtion is fixed");
	prm.declare_entry("top fixed value file","average_borehole_A_05_09.txt",
			  Patterns::Anything(),
			  "file containing values for the top boundary"
			  "in case of fixed conditions");
      }
      prm.leave_subsection();
    }
  
  template <int dim>
    void AllParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      //mesh_filename = prm.get("mesh");
      
      prm.enter_subsection("time stepping");
      {
	time_step           = prm.get_double ("time step");
	timestep_number_max = prm.get_integer("timestep number max");
	theta               = prm.get_double ("theta scheme value");
      }
      prm.leave_subsection();


      prm.enter_subsection("geometric data");
      {
	domain_size          = prm.get_double  ("domain size");
	insulation_thickness = prm.get_double  ("insulation thickness");
	insulation_depth     = prm.get_double  ("insulation depth");
	refinement_level     = prm.get_integer ("refinement level");
      }
      prm.leave_subsection();


      prm.enter_subsection("material data");
      {
	soil_thermal_conductivity         = prm.get_double ("soil thermal conductivity");
	soil_density                      = prm.get_double ("soil density");
	soil_specific_heat_capacity       = prm.get_double ("soil specific heat capacity");
	insulation_thermal_conductivity   = prm.get_double ("insulation thermal conductivity");
	insulation_density                = prm.get_double ("insulation density");
	insulation_specific_heat_capacity = prm.get_double ("insulation specific heat capacity");

      }
      prm.leave_subsection();


      prm.enter_subsection("boundary conditions");
      {
	fixed_at_bottom      = prm.get_bool   ("fixed at bottom");
	bottom_fixed_value   = prm.get_double ("bottom fixed value");
	fixed_at_top         = prm.get_bool   ("fixed at top");
	top_fixed_value_file = prm.get        ("top fixed value file");
      }
      prm.leave_subsection();

    }
}
