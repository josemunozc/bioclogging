

namespace Parameters
{  
template <int dim>
struct AllParameters
{
	AllParameters ();

	unsigned int timestep_number_max;
	double time_step;
	double theta;
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

	double bottom_fixed_value;
	double top_fixed_value;
	double initial_condition;

	bool fixed_at_bottom;
	bool fixed_at_top;
	bool use_mesh_file;
	bool lumped_matrix;

	std::string mesh_filename;
	std::string moisture_transport_equation;
	std::string hydraulic_properties;

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


	prm.enter_subsection("boundary conditions");
	{
		prm.declare_entry("initial condition", "0.",
				Patterns::Double(),
				"set homogeneous initial value for domain");
		prm.declare_entry("fixed at bottom", "false",
				Patterns::Bool(),
				"if true, the bottom boundary condition is fixed");
		prm.declare_entry("bottom fixed value", "10.",
				Patterns::Double(),
				"value at bottom boundary for fixed conditions");
		prm.declare_entry("fixed at top", "true",
				Patterns::Bool(),
				"if true, the top boundary condition is fixed");
		prm.declare_entry("top fixed value","0.",
				Patterns::Double(),
				"value at top boundary for fixed conditions");
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
	}
	prm.leave_subsection();
}

template <int dim>
void AllParameters<dim>::parse_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("time stepping");
	{
		time_step           = prm.get_double ("time step");
		timestep_number_max = prm.get_integer("timestep number max");
		theta               = prm.get_double ("theta scheme value");
	}
	prm.leave_subsection();


	prm.enter_subsection("geometric data");
	{
		use_mesh_file        = prm.get_bool    ("use mesh file");
		mesh_filename        = prm.get         ("mesh filename");
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
		initial_condition =prm.get_double("initial condition");
		fixed_at_bottom   =prm.get_bool  ("fixed at bottom");
		bottom_fixed_value=prm.get_double("bottom fixed value");
		fixed_at_top      =prm.get_bool  ("fixed at top");
		top_fixed_value   =prm.get_double("top fixed value");
	}
	prm.leave_subsection();

	prm.enter_subsection("equations");
	{
		moisture_transport_equation=prm.get("moisture transport");
		hydraulic_properties       =prm.get("hydraulic properties");
		lumped_matrix              =prm.get_bool("lumped matrix (qtrapez)");
	}
	prm.leave_subsection();
}
}
