/*!
 * @file diffusion_problem_ms.hpp
 * @brief Contains implementation of the main object for multiscale FEM.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_MS_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_MS_HPP_

/* ***************
 * Deal.ii
 * ***************
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/cell_id.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
/* ***************
 * Deal.ii
 * ***************
 * */

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

// My Headers
#include "config.h"
#include "matrix_coeff.hpp"
#include "right_hand_side.hpp"
#include "neumann_bc.hpp"
#include "dirichlet_bc.hpp"
#include "diffusion_problem_basis.hpp"

/*!
 * @namespace DiffusionProblem
 * @brief Contains implementation of the main object
 * and all functions to solve a
 * Dirichlet-Neumann problem on a unit square.
 */
namespace DiffusionProblem
{
using namespace dealii;

/*!
 * @class DiffusionProblemMultiscale
 * @brief Main class to solve
 * Dirichlet-Neumann problem on a unit square with
 * multiscale FEM.
 */
template <int dim>
class DiffusionProblemMultiscale
{
public:
	DiffusionProblemMultiscale (unsigned int n_refine, unsigned int n_refine_local);
	void run ();

private:
	void make_grid ();
	void initialize_and_compute_basis ();
	void setup_system ();
	void assemble_system ();
	void solve_iterative ();
	void send_global_weights_to_cell ();

	void output_global_coarse () const;
	void output_global_fine ();

	MPI_Comm mpi_communicator;

	parallel::distributed::Triangulation<dim> triangulation;

	FE_Q<dim>            			fe;
	DoFHandler<dim>      			dof_handler;

	IndexSet 	locally_owned_dofs;
	IndexSet 	locally_relevant_dofs;

	AffineConstraints<double> 		constraints;

	/*!
	 *
	 */
	LA::MPI::SparseMatrix 		system_matrix;

	/*!
	 * Solution vector containing weights at the dofs.
	 */
	LA::MPI::Vector       		locally_relevant_solution;

	/*!
	 * Contains all parts of the right-hand side needed to
	 * solve the linear system.
	 */
	LA::MPI::Vector       		system_rhs;

	ConditionalOStream 		pcout;
	TimerOutput        		computing_timer;

	/*!
	 * Number of global refinements.
	 */
	const unsigned int n_refine;

	/*!
	 * Number of local refinements.
	 */
	const unsigned int n_refine_local;

	/*!
	 * STL Vector holding basis functions for each coarse cell.
	 */
	std::map<CellId, DiffusionProblemBasis<dim>> 	cell_basis_map;
};


/*!
 * Constructor.
 */
template <int dim>
DiffusionProblemMultiscale<dim>::DiffusionProblemMultiscale (unsigned int n_refine,
		unsigned int n_refine_local)
:
mpi_communicator(MPI_COMM_WORLD),
triangulation(mpi_communicator,
			  typename Triangulation<dim>::MeshSmoothing(
				Triangulation<dim>::smoothing_on_refinement |
				Triangulation<dim>::smoothing_on_coarsening)),
fe (1),
dof_handler (triangulation),
pcout(std::cout,
	  (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
computing_timer(mpi_communicator,
				pcout,
				TimerOutput::summary,
				TimerOutput::wall_times),
n_refine(n_refine),
n_refine_local(n_refine_local),
cell_basis_map()
{}


/*!
 * Set all relevant data to local basis object and initialize the basis fully. Then compute.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::initialize_and_compute_basis ()
{
	TimerOutput::Scope t(computing_timer, "basis initialization and computation");

	typename Triangulation<dim>::active_cell_iterator
									cell = dof_handler.begin_active(),
									endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if (cell->is_locally_owned())
		{
			DiffusionProblemBasis<dim> current_cell_problem(n_refine_local, cell);
			CellId current_cell_id(cell->id());

			std::pair<typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator, bool > result;
			result = cell_basis_map.insert(std::make_pair(cell->id(),
					current_cell_problem));

			Assert(result.second,
					ExcMessage ("Insertion of local basis problem into std::map failed. "
							"Problem with copy constructor?"));

			typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator it_basis = cell_basis_map.find(cell->id());

			(it_basis->second).run();
		}
	} // end ++cell
}

/*!
 * @brief Set up the grid with a certain number of refinements.
 *
 * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
 * numbered form \f$1,\dots,2\rm{dim}\f$.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::make_grid ()
{
	TimerOutput::Scope t(computing_timer, "global mesh generation");

	GridGenerator::hyper_cube (triangulation, 0, 1, /* colorize */ true);

	triangulation.refine_global (n_refine);

	pcout << "Number of active global cells: "
			<< triangulation.n_active_cells()
			<< std::endl;
}


/*!
 * @brief Setup sparsity pattern and system matrix.
 *
 * Compute sparsity pattern and reserve memory for the sparse system matrix
 * and a number of right-hand side vectors. Also build a constraint object
 * to take care of Dirichlet boundary conditions.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::setup_system ()
{
	TimerOutput::Scope t(computing_timer, "global system setup");

	dof_handler.distribute_dofs(fe);

	locally_owned_dofs = dof_handler.locally_owned_dofs();
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

	locally_relevant_solution.reinit(locally_owned_dofs,
								 locally_relevant_dofs,
								 mpi_communicator);

	system_rhs.reinit(locally_owned_dofs, mpi_communicator);

	constraints.clear();
	constraints.reinit(locally_relevant_dofs);

	DoFTools::make_hanging_node_constraints(dof_handler, constraints);

	// Set up Dirichlet boundary conditions.
	const Coefficients::DirichletBC<dim> dirichlet_bc;
	for (unsigned int i = 0; i<dim; ++i)
	{
		VectorTools::interpolate_boundary_values(dof_handler,
													/*boundary id*/ 2*i, // only even boundary id
													dirichlet_bc,
													constraints);
	}

	constraints.close();

	DynamicSparsityPattern dsp(locally_relevant_dofs);
	DoFTools::make_sparsity_pattern (dof_handler,
			dsp,
			constraints,
			/*keep_constrained_dofs =*/ true);
	SparsityTools::distribute_sparsity_pattern(dsp,
			dof_handler.n_locally_owned_dofs_per_processor(),
			mpi_communicator,
			locally_relevant_dofs);

	system_matrix.reinit(locally_owned_dofs,
			locally_owned_dofs,
			dsp,
			mpi_communicator);
}


/*!
 * @brief Assemble the system matrix and the static right hand side.
 *
 * Assembly routine to build the time-independent (static) part.
 * Neumann boundary conditions will be put on edges/faces
 * with odd number. Constraints are not applied here yet.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::assemble_system ()
{
	TimerOutput::Scope t(computing_timer, "global multiscale assembly");

	QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

	FEFaceValues<dim> 	fe_face_values(fe,
										face_quadrature_formula,
										update_values | update_quadrature_points |
										update_normal_vectors |
										update_JxW_values);

	const unsigned int   	dofs_per_cell = fe.dofs_per_cell;
	const unsigned int 		n_face_q_points = face_quadrature_formula.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	/*
	 * Neumann BCs and vector to store the values.
	 */
	const Coefficients::NeumannBC<dim> 	neumann_bc;
	std::vector<double>  	neumann_values(n_face_q_points);

	/*
	 * Integration over cells.
	 */
	for (const auto &cell: dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator it_basis = cell_basis_map.find(cell->id());

			cell_matrix = 0;
			cell_rhs = 0;

			cell_matrix = (it_basis->second).get_global_element_matrix ();
			cell_rhs = (it_basis->second).get_global_element_rhs ();

			/*
			 * Boundary integral for Neumann values for odd boundary_id.
			 */
			for (unsigned int face_number = 0;
				 face_number < GeometryInfo<dim>::faces_per_cell;
				 ++face_number)
			{
				if (cell->face(face_number)->at_boundary() &&
						(
							(cell->face(face_number)->boundary_id() == 1) ||
							(cell->face(face_number)->boundary_id() == 3) ||
							(cell->face(face_number)->boundary_id() == 5)
						)
					)
				{
					fe_face_values.reinit(cell, face_number);

					// Fill in values at this particular face.
					neumann_bc.value_list(fe_face_values.get_quadrature_points(),
											   neumann_values);

					for (unsigned int q_face_point = 0; q_face_point < n_face_q_points; ++q_face_point)
					{
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
						{
							cell_rhs(i) += neumann_values[q_face_point] // g(x_q)
											* fe_face_values.shape_value(i, q_face_point) // phi_i(x_q)
											* fe_face_values.JxW(q_face_point); // dS
						} // end ++i
					} // end ++q_face_point
				} // end if
			} // end ++face_number


			// get global indices
			cell->get_dof_indices (local_dof_indices);
			/*
			 * Now add the cell matrix and rhs to the right spots
			 * in the global matrix and global rhs. Constraints will
			 * be taken care of later.
			 */
			constraints.distribute_local_to_global(cell_matrix,
													 cell_rhs,
													 local_dof_indices,
													 system_matrix,
													 system_rhs);
		}
	} // end ++cell

	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);
}


/*!
 * @brief Iterative solver.
 *
 * CG-based solver with SSOR-preconditioning.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::solve_iterative ()
{
	TimerOutput::Scope t(computing_timer, "global iterative solver");

	LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
			mpi_communicator);

	SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

	#ifdef USE_PETSC_LA
		LA::SolverCG solver(solver_control, mpi_communicator);
	#else
		LA::SolverCG solver(solver_control);
	#endif

		LA::MPI::PreconditionAMG preconditioner;
		LA::MPI::PreconditionAMG::AdditionalData data;

	#ifdef USE_PETSC_LA
		data.symmetric_operator = true;
	#else
		/* Trilinos defaults are good */
	#endif

		preconditioner.initialize(system_matrix, data);

		solver.solve(system_matrix,
					 completely_distributed_solution,
					 system_rhs,
					 preconditioner);

		pcout << "   Solved in " << solver_control.last_step() << " iterations."
			  << std::endl;

		constraints.distribute(completely_distributed_solution);
		locally_relevant_solution = completely_distributed_solution;
}


/*!
 * @brief Send coarse weights to corresponding local cell.
 *
 * After the coarse (global) weights have been computed they
 * must be set to the local basis object and stored there.
 * This is necessary to write the local multiscale solution.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::send_global_weights_to_cell ()
{
	// For each cell we get dofs_per_cell values
	const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
	std::vector<types::global_dof_index> 	local_dof_indices (dofs_per_cell);

	// active cell iterator
	typename DoFHandler<dim>::active_cell_iterator
								cell = dof_handler.begin_active (),
								endc = dof_handler.end ();
	for (; cell!=endc; ++cell)
	{
		if (cell->is_locally_owned())
		{
			cell->get_dof_indices (local_dof_indices);
			std::vector<double> extracted_weights (dofs_per_cell, 0);
			locally_relevant_solution.extract_subvector_to (local_dof_indices, extracted_weights);

			typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator it_basis = cell_basis_map.find(cell->id());
			(it_basis->second).set_global_weights (extracted_weights);
		}
	} // end ++cell
}


/*!
 * @brief Write coarse solution to disk.
 *
 * Write results for coarse solution to disk in vtu-format.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::output_global_coarse () const
{
	std::string filename = (dim == 2 ?
									"solution-ms_coarse-2d" :
									"solution-ms_coarse-3d" );

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(locally_relevant_solution, "u");

	Vector<float> subdomain(triangulation.n_active_cells());
	for (unsigned int i = 0; i < subdomain.size(); ++i)
	{
	  subdomain(i) = triangulation.locally_owned_subdomain();
	}

	data_out.add_data_vector(subdomain, "subdomain");

	data_out.build_patches ();

	std::string filename_slave (filename);
	filename_slave += "_refinements-" + Utilities::int_to_string(n_refine, 1)
				+ "." + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4)
				+ ".vtu";

	std::ofstream output (filename_slave.c_str());
	data_out.write_vtu (output);


	if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
	{
		std::vector<std::string> file_list;
		for (unsigned int i = 0;
		   i < Utilities::MPI::n_mpi_processes(mpi_communicator);
		   ++i)
		{
			file_list.push_back(filename
					+ "_refinements-" + Utilities::int_to_string(n_refine, 1)
					+ "." + Utilities::int_to_string(i, 4)
					+ ".vtu");
		}

		std::string filename_master(filename);

		filename_master += "_refinements-" + Utilities::int_to_string(n_refine, 1) + ".pvtu";

		std::ofstream master_output(filename_master.c_str());
		data_out.write_pvtu_record(master_output, file_list);
	}
}


/*!
 * Write all local multiscale solution (threaded) and
 * a global pvtu-record.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::output_global_fine ()
{
	typename Triangulation<dim>::active_cell_iterator
										cell = dof_handler.begin_active(),
										endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if (cell->is_locally_owned())
		{
			typename std::map<CellId, DiffusionProblemBasis<dim>>::iterator it_basis = cell_basis_map.find(cell->id());
			// Get the global file name
			(it_basis->second).output_global_solution_in_cell ();
		}
	} // end ++cell



//	if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
//	{
//		DataOut<dim> data_out;
//		data_out.attach_dof_handler (dof_handler);
//
//		// Names of solution components
//		data_out.add_data_vector (solution, "solution");
//
//		std::vector<std::string> file_list;
//		for (unsigned int i = 0;
//		   i < Utilities::MPI::n_mpi_processes(mpi_communicator);
//		   ++i)
//		{
//			MPI_Send()
//			file_list.push_back(received_name);
//		}
//
//		std::string filename_master = (dim == 2 ?
//					"solution-ms_fine-2d" :
//					"solution-ms_fine-3d");
//		filename_master += ".pvtu";
//
//		std::ofstream master_output(filename_master.c_str ());
//		data_out.write_pvtu_record(master_output, filenames_on_cell);
//	}
//	else // send local filename for list
//	{
//		unsigned int str_length =
//		char* local_filename;
//	}
}


/*!
 * @brief Run function of the object.
 *
 * Run the computation after object is built.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::run ()
{
	pcout << std::endl
			<< "==========================================="
			<< std::endl
			<< "Solving >> MULTISCALE << problem in "
			<< dim
			<< "D."
			<< std::endl;

	pcout << "Running with "
	#ifdef USE_PETSC_LA
			  << "PETSc"
	#else
			  << "Trilinos"
	#endif
			  << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
			  << " MPI rank(s)..." << std::endl;

	make_grid ();

	setup_system ();

	initialize_and_compute_basis ();

	assemble_system ();

	// Now solve
	solve_iterative ();

	send_global_weights_to_cell ();

	if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
	{
		TimerOutput::Scope t(computing_timer, "coarse output vtu");
		output_global_coarse ();
	}
	if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
	{
		TimerOutput::Scope t(computing_timer, "fine output vtu");
		output_global_fine ();
	}

	computing_timer.print_summary();
	computing_timer.reset();

	pcout << std::endl
			<< "===========================================" << std::endl;
}

} // end namespace DiffusionProblem


#endif /* INCLUDE_DIFFUSION_PROBLEM_MS_HPP_ */
