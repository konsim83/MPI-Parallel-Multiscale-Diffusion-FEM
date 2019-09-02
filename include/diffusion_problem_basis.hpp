/*!
 * @file diffusion_problem_basis.hpp
 * @brief Contains implementation of multiscale basis functions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_

// Deal.ii
#include <deal.II/base/mpi.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

// STL
#include <cmath>
#include <fstream>
#include <iostream>

// My Headers
#include "config.h"
#include "matrix_coeff.hpp"
#include "right_hand_side.hpp"
#include "neumann_bc.hpp"
#include "basis_q1.hpp"

/*!
 * @namespace DiffusionProblem
 * @brief Contains implementation of the main object
 * and all functions to solve a time-independent
 * Dirichlet-Neumann problem on a unit square.
 */
namespace DiffusionProblem
{
using namespace dealii;

/*!
 * @class DiffusionProblemBasis
 * @brief Main class to solve for time-independent
 * multiscale basis functions (Dirichlet problem) on a
 * given coarse quadrilateral cell without oversampling.
 */
template <int dim>
class DiffusionProblemBasis
{
public:
	DiffusionProblemBasis () = delete;
	DiffusionProblemBasis (unsigned int n_refine_local,
					typename Triangulation<dim>::active_cell_iterator& global_cell,
					unsigned int local_subdomain,
					MPI_Comm mpi_communicator);
	DiffusionProblemBasis (const DiffusionProblemBasis<dim> &X);
	void run ();

	void output_global_solution_in_cell () const;

	const FullMatrix<double>& get_global_element_matrix () const;
	const Vector<double>& get_global_element_rhs () const;
	const std::string& get_filename_global ();

	void set_global_weights (const std::vector<double> &global_weights);
	void set_output_flag (bool flag);

private:
	void make_grid ();
	void setup_system ();
	void assemble_system ();
	void assemble_global_element_matrix ();
	void solve_iterative (unsigned int index_basis);
	void output_basis () const;

	void set_filename_global ();

	MPI_Comm mpi_communicator;

	Triangulation<dim>   			triangulation;
	FE_Q<dim>            			fe;
	DoFHandler<dim>      			dof_handler;

	std::vector<AffineConstraints<double>> 		constraints_vector;
	std::vector<Point<dim>> 					corner_points;

	SparsityPattern      			sparsity_pattern;
	SparseMatrix<double> 			diffusion_matrix;
	SparseMatrix<double> 			system_matrix;

	std::string	filename_global;

	/*!
	 * Solution vector.
	 */
	std::vector<Vector<double>>		solution_vector;

	/*!
	 * Contains the right-hand side.
	 */
	Vector<double>       			global_rhs; // this is only for the global assembly (speed-up)

	/*!
	 * Contains all parts of the right-hand side needed to
	 * solve the linear system..
	 */
	Vector<double>					system_rhs;

	/*!
	 * Holds global multiscale element matrix.
	 */
	FullMatrix<double>   		global_element_matrix;
	bool is_built_global_element_matrix;

	/*!
	 * Holds global multiscale element right-hand side.
	 */
	Vector<double>   			global_element_rhs;

	/*!
	 * Weights of multiscale basis functions.
	 */
	std::vector<double> 		global_weights;
	bool is_set_global_weights;

	/*!
	 * Global solution
	 */
	Vector<double>			global_solution;

	/*!
	 * Number of local refinements.
	 */
	const unsigned int n_refine_local;

	/*!
	 * Global cell number.
	 */
	const CellId global_cell_id;

	/*!
	 * Global subdomain number.
	 */
	const unsigned int local_subdomain;

	/*!
	 * Object carries set of local \f$Q_1\f$-basis functions.
	 */
	Coefficients::BasisQ1<dim> basis_q1;

	/*!
	 * Write basis functions as vtu.
	 */
	bool output_flag;

	/*!
	 * Text output on runtime.
	 */
	bool verbose;
};


/*!
 * Default constructor.
 */
template <int dim>
DiffusionProblemBasis<dim>::DiffusionProblemBasis (unsigned int n_refine_local,
		typename Triangulation<dim>::active_cell_iterator& global_cell,
		unsigned int local_subdomain,
		MPI_Comm mpi_communicator)
:
mpi_communicator(mpi_communicator),
fe (1),
dof_handler (triangulation),
constraints_vector (GeometryInfo<dim>::vertices_per_cell),
corner_points (GeometryInfo<dim>::vertices_per_cell),
filename_global (""),
solution_vector(GeometryInfo<dim>::vertices_per_cell),
global_element_matrix (fe.dofs_per_cell,
	fe.dofs_per_cell),
is_built_global_element_matrix (false),
global_element_rhs (fe.dofs_per_cell),
global_weights (fe.dofs_per_cell, 0),
is_set_global_weights (false),
n_refine_local (n_refine_local),
global_cell_id (global_cell->id()),
local_subdomain(local_subdomain),
basis_q1 (global_cell),
output_flag (false),
verbose(false)
{
	// set corner points
	for (unsigned int vertex_n=0;
			 vertex_n<GeometryInfo<dim>::vertices_per_cell;
			 ++vertex_n)
	{
		corner_points[vertex_n] = global_cell->vertex(vertex_n);
	}
}


/*!
 * Copy constructor.
 */
template <int dim>
DiffusionProblemBasis<dim>::DiffusionProblemBasis (const DiffusionProblemBasis<dim> &X)
:
//triangulation(X.triangulation), // only possible if object is empty
mpi_communicator(X.mpi_communicator),
fe (X.fe),
dof_handler (triangulation), // must be constructed deliberately
constraints_vector (X.constraints_vector),
corner_points (X.corner_points),
sparsity_pattern (X.sparsity_pattern), // only possible if object is empty
diffusion_matrix (X.diffusion_matrix), // only possible if object is empty
system_matrix (X.system_matrix), // only possible if object is empty
filename_global (X.filename_global),
solution_vector (X.solution_vector),
global_rhs (X.global_rhs),
system_rhs (X.system_rhs),
global_element_matrix (X.global_element_matrix),
is_built_global_element_matrix (X.is_built_global_element_matrix),
global_element_rhs (X.global_element_rhs),
global_weights (X.global_weights),
is_set_global_weights (X.is_set_global_weights),
global_solution (X.global_solution),
n_refine_local (X.n_refine_local),
global_cell_id (X.global_cell_id),
local_subdomain(X.local_subdomain),
basis_q1 (X.basis_q1),
output_flag (X.output_flag),
verbose(X.verbose)
{
//	triangulation.copy_triangulation (X.triangulation);
}


/*!
 * @brief Set up the grid with a certain number of refinements.
 *
 * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
 * numbered form \f$1,\dots,2\rm{dim}\f$.
 */
template <int dim>
void DiffusionProblemBasis<dim>::make_grid ()
{
	GridGenerator::general_cell(triangulation, corner_points, /* colorize faces */ false);

	triangulation.refine_global (n_refine_local);
}


/*!
 * @brief Setup sparsity pattern and system matrix.
 *
 * Compute sparsity pattern and reserve memory for the sparse system matrix
 * and a number of right-hand side vectors. Also build a constraint object
 * to take care of Dirichlet boundary conditions.
 */
template <int dim>
void DiffusionProblemBasis<dim>::setup_system ()
{
	dof_handler.distribute_dofs (fe);

	if (verbose)
		std::cout << "Global cell id  "
				<< global_cell_id.to_string()
				<< ":   "
				<< triangulation.n_active_cells() << " active fine cells --- "
				<< dof_handler.n_dofs() << " subgrid dof"
				<< std::endl;

	/*
	 * Set up Dirichlet boundary conditions and sparsity pattern.
	 */
	DynamicSparsityPattern dsp(dof_handler.n_dofs());

	for (unsigned int index_basis = 0;
			index_basis<GeometryInfo<dim>::vertices_per_cell;
			++index_basis)
	{
		basis_q1.set_index (index_basis);

		constraints_vector[index_basis].clear();
		DoFTools::make_hanging_node_constraints(dof_handler, constraints_vector[index_basis]);

		VectorTools::interpolate_boundary_values(dof_handler,
													/*boundary id*/ 0,
													basis_q1,
													constraints_vector[index_basis]);
		constraints_vector[index_basis].close();
	}

	DoFTools::make_sparsity_pattern (dof_handler,
									dsp,
									constraints_vector[0], // sparsity pattern is the same for each basis
									/*keep_constrained_dofs =*/ true); // for time stepping this is essential to be true
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit (sparsity_pattern);
	diffusion_matrix.reinit (sparsity_pattern);

	for (unsigned int index_basis = 0;
			index_basis<GeometryInfo<dim>::vertices_per_cell;
			++index_basis)
	{
		solution_vector[index_basis].reinit (dof_handler.n_dofs());
	}
	system_rhs.reinit (dof_handler.n_dofs());
	global_rhs.reinit (dof_handler.n_dofs());
}


/*!
 * @brief Assemble the system matrix and the static right hand side.
 *
 * Assembly routine to build the time-independent (static) part.
 * Neumann boundary conditions will be put on edges/faces
 * with odd number. Constraints are not applied here yet.
 */
template <int dim>
void DiffusionProblemBasis<dim>::assemble_system ()
{
	QGauss<dim>  quadrature_formula(fe.degree + 1);

	FEValues<dim> 	fe_values (fe, quadrature_formula,
								update_values    |  update_gradients |
								update_quadrature_points  |  update_JxW_values);

	const unsigned int   	dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   	n_q_points    = quadrature_formula.size();

	FullMatrix<double>   cell_diffusion_matrix (dofs_per_cell, dofs_per_cell);
	FullMatrix<double>   cell_mass_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	/*
	 * Matrix coefficient and vector to store the values.
	 */
	const Coefficients::MatrixCoeff<dim> 		matrix_coeff;
	std::vector<Tensor<2,dim>> 	matrix_coeff_values(n_q_points);

	/*
	 * Right hand side and vector to store the values.
	 */
	const Coefficients::RightHandSide<dim> 	right_hand_side;
	std::vector<double>      	rhs_values(n_q_points);

	/*
	 * Integration over cells.
	 */
	for (const auto &cell: dof_handler.active_cell_iterators())
	{
		cell_diffusion_matrix = 0;
		cell_rhs = 0;

		fe_values.reinit (cell);

		// Now actually fill with values.
		matrix_coeff.value_list(fe_values.get_quadrature_points (),
						  	  	  matrix_coeff_values);
		right_hand_side.value_list(fe_values.get_quadrature_points(),
									   rhs_values);

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{

					cell_diffusion_matrix(i,j) += fe_values.shape_grad(i,q_index) *
										 matrix_coeff_values[q_index] *
										 fe_values.shape_grad(j,q_index) *
										 fe_values.JxW(q_index);
				} // end ++j

				cell_rhs(i) += fe_values.shape_value(i,q_index) *
								   rhs_values[q_index] *
								   fe_values.JxW(q_index);
			} // end ++i
		} // end ++q_index

		// get global indices
		cell->get_dof_indices (local_dof_indices);
		/*
		 * Now add the cell matrix and rhs to the right spots
		 * in the global matrix and global rhs. Constraints will
		 * be taken care of later.
		 */
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
			{
				diffusion_matrix.add(local_dof_indices[i],
							local_dof_indices[j],
							cell_diffusion_matrix(i, j));
			}
			global_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	} // end ++cell
}


/*!
 *
 */
template <int dim>
void
DiffusionProblemBasis<dim>::assemble_global_element_matrix ()
{
	// First, reset.
	global_element_matrix = 0;

	// Get lengths of tmp vectors for assembly
	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

	Vector<double>			tmp (dof_handler.n_dofs());

	// This assembles the local contribution to the global global matrix
	// with an algebraic trick. It uses the local system matrix stored in
	// the respective basis object.
	for (unsigned int i_test=0;
			i_test < dofs_per_cell;
			++i_test)
	{
		// set an alias name
		const Vector<double>& test_vec = solution_vector[i_test];

		for (unsigned int i_trial=0;
				i_trial<dofs_per_cell;
				++i_trial)
		{
			// set an alias name
			const Vector<double>& trial_vec = solution_vector[i_trial];

			// tmp = system_matrix*trial_vec
			diffusion_matrix.vmult(tmp, trial_vec);

			// global_element_diffusion_matrix = test_vec*tmp
			global_element_matrix(i_test,i_trial) += (test_vec * tmp);

			// reset
			tmp = 0;
		} // end for i_trial

		global_element_rhs(i_test) += test_vec * global_rhs;

	} // end for i_test

	is_built_global_element_matrix = true;
}


/*!
 * @brief Iterative solver.
 *
 * CG-based solver with SSOR-preconditioning.
 */
template <int dim>
void DiffusionProblemBasis<dim>::solve_iterative (unsigned int index_basis)
{
	SolverControl           solver_control (1000, 1e-12);
	SolverCG<>              solver (solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);

	solver.solve (system_matrix,
				solution_vector[index_basis],
				system_rhs,
				preconditioner);

	constraints_vector[index_basis].distribute (solution_vector[index_basis]);

	if (verbose)
		std::cout << "   "
				<< "(cell   "
				<< global_cell_id.to_string()
				<< ") "
				<< "(basis   "
				<< index_basis
				<< ")   "
				<< solver_control.last_step()
				<< " fine CG iterations needed to obtain convergence."
				<< std::endl;
}


/*!
 * Return the multiscale element matrix produced
 * from local basis functions.
 */
template <int dim>
const FullMatrix<double>&
DiffusionProblemBasis<dim>::get_global_element_matrix () const
{
	return global_element_matrix;
}


/*!
 * Get the right hand-side that was locally assembled
 * to speed up the global assembly.
 */
template <int dim>
const Vector<double>&
DiffusionProblemBasis<dim>::get_global_element_rhs () const
{
	return global_element_rhs;
}

/*!
 * Return filename for local pvtu record.
 */
template <int dim>
const std::string&
DiffusionProblemBasis<dim>::get_filename_global ()
{
	return filename_global;
}


/*!
 * Set the output flag to write basis functions to disk as vtu.
 * @param flag
 */
template <int dim>
void
DiffusionProblemBasis<dim>::set_output_flag (bool flag)
{
	output_flag = flag;
}


/*!
 * @brief Set global weights.
 * @param weights
 *
 * The coarse weights of the global solution determine
 * the local multiscale solution. They must be computed
 * and then set locally to write an output.
 */
template <int dim>
void
DiffusionProblemBasis<dim>::set_global_weights (const std::vector<double> &weights)
{
	// Copy assignment of global weights
	global_weights = weights;

	// reinitialize the global solution on this cell
	global_solution.reinit (dof_handler.n_dofs());

	const unsigned int dofs_per_cell	= fe.n_dofs_per_cell();

	// Set global solution using the weights and the local basis.
	for (unsigned int index_basis=0;
			index_basis<dofs_per_cell;
			++index_basis)
	{
		// global_solution = 1*global_solution + global_weights[index_basis]*solution_vector[index_basis]
		global_solution.sadd (1, global_weights[index_basis], solution_vector[index_basis]);
	}

	is_set_global_weights = true;
}


/*!
 * Define the gloabl filename for pvtu-file in global output.
 */
template <int dim>
void
DiffusionProblemBasis<dim>::set_filename_global ()
{
	filename_global += (dim == 2 ?
			"solution-ms_fine-2d" :
			"solution-ms_fine-3d");

	filename_global += "." + Utilities::int_to_string(local_subdomain, 5);
	filename_global += ".cell-" + global_cell_id.to_string() + ".vtu";
}


/*!
 * @brief Write basis results to disk.
 *
 * Write basis results to disk in vtu-format.
 */
template <int dim>
void
DiffusionProblemBasis<dim>::output_basis () const
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);
	for (unsigned int index_basis=0;
				index_basis<GeometryInfo<dim>::vertices_per_cell;
				++index_basis)
	{
		data_out.add_data_vector (solution_vector[index_basis], "basis_" + Utilities::int_to_string(index_basis, 1));
	}
	data_out.build_patches ();

	std::string filename = "basis";
	filename += "." + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 5);
	filename += ".cell-" + global_cell_id.to_string();
	filename += ".vtu";

	std::ofstream output (dim == 2 ?
					"2d-" + filename :
					"3d-" + filename);

	data_out.write_vtu (output);
}


/*!
 * Write out global solution in cell.
 */
template <int dim>
void
DiffusionProblemBasis<dim>::output_global_solution_in_cell () const
{
	Assert (is_set_global_weights,
				ExcMessage ("Global weights must be set first."));

	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (global_solution, "solution");
	data_out.build_patches ();

	std::ofstream output (filename_global.c_str());
	data_out.write_vtu (output);
}


/*!
 * @brief Run function of the object.
 *
 * Run the computation after object is built.
 */
template <int dim>
void DiffusionProblemBasis<dim>::run ()
{
	make_grid ();

	setup_system ();

	assemble_system ();

	set_filename_global ();

	for (unsigned int index_basis=0;
			index_basis<GeometryInfo<dim>::vertices_per_cell;
			++index_basis)
	{
		// reset everything
		system_rhs.reinit(solution_vector[index_basis].size());
		system_matrix.reinit (sparsity_pattern);

		system_matrix.copy_from(diffusion_matrix);

		// Now take care of constraints
		constraints_vector[index_basis].condense(system_matrix, system_rhs);

		// Now solve
		solve_iterative (index_basis);
	}

	assemble_global_element_matrix ();

	if (output_flag)
		output_basis ();
}

} // end namespace DiffusionProblem

#endif /* INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_ */
