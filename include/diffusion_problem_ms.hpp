/*!
 * @file diffusion_problem_ms.hpp
 * @brief Contains implementation of the main object for multiscale FEM.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_MS_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_MS_HPP_

// Deal.ii
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>

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
#include <vector>

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
	// *********************************************
	// This is for threading the basis computation
	struct BasisScratchData
	{
		BasisScratchData () {}; // No implementation
		BasisScratchData (const BasisScratchData& /*scratch_data*/) {}; // No implementation
	};

	struct BasisCopyData
	{

	};

	void compute_local_basis(const typename std::vector<DiffusionProblemBasis<dim>>::iterator &it_basis,
								BasisScratchData	&scratch_data,
								BasisCopyData	&copy_data);
	void output_local_solution(const typename std::vector<DiffusionProblemBasis<dim>>::iterator &it_basis,
								BasisScratchData	&scratch_data,
								BasisCopyData	&copy_data);
	void fake_copy (const BasisCopyData&) {}; // No implementation
	// *********************************************

	void make_grid ();
	void initialize_basis_problem ();
	void compute_basis ();
	void setup_system ();
	void assemble_system ();
	void solve_iterative ();

	void send_global_weights_to_cell ();

	void output_global_coarse () const;
	void output_global_fine ();

	Triangulation<dim>   			triangulation;
	FE_Q<dim>            			fe;
	DoFHandler<dim>      			dof_handler;

	AffineConstraints<double> 		constraints;

	SparsityPattern      			sparsity_pattern;
	SparseMatrix<double> 			system_matrix;

	/*!
	 * Solution vector containing weights at the dofs.
	 */
	Vector<double>       			solution;

	/*!
	 * Contains all parts of the right-hand side needed to
	 * solve the linear system.
	 */
	Vector<double>       			system_rhs;

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
	std::vector<DiffusionProblemBasis<dim>> 	cell_basis_vector;
};


/*!
 * Constructor.
 */
template <int dim>
DiffusionProblemMultiscale<dim>::DiffusionProblemMultiscale (unsigned int n_refine, unsigned int n_refine_local) :
  fe (1),
  dof_handler (triangulation),
  n_refine(n_refine),
  n_refine_local(n_refine_local),
  cell_basis_vector(std::pow(2,dim*n_refine))
{}


/*!
 * Set all relevant data to local basis object and initialize the basis fully.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::initialize_basis_problem ()
{
	// First set up all cell problems serially
	typename Triangulation<dim>::active_cell_iterator
									cell = dof_handler.begin_active(),
									endc = dof_handler.end();
	unsigned int cell_number = 0;
	for (; cell!=endc; ++cell)
	{
		cell_basis_vector[cell_number].set_n_local_refinements (n_refine_local);
		cell_basis_vector[cell_number].set_cell_data (cell, cell_number);
		cell_basis_vector[cell_number].set_basis_data ();

		if (cell_number==0)
			cell_basis_vector[cell_number].set_output_flag (true);

		++cell_number;
	}
}

/*!
 * @brief Function pre-computes basis functions.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::compute_basis ()
{
	// Now run them in threads
	typename std::vector<DiffusionProblemBasis<dim>>::iterator
										it_basis = cell_basis_vector.begin (),
										it_endbasis = cell_basis_vector.end ();
	WorkStream::run(it_basis,
					it_endbasis,
					*this,
					&DiffusionProblemMultiscale<dim>::compute_local_basis,
					&DiffusionProblemMultiscale<dim>::fake_copy,
					BasisScratchData(),
					BasisCopyData());
}


/*!
 * Pre-compute the local basis. This function
 * is only used for threading.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::compute_local_basis(const typename
														std::vector<DiffusionProblemBasis<dim>>::iterator &it_basis,
														BasisScratchData&,
														BasisCopyData&)
{
	it_basis->run ();
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
	GridGenerator::hyper_cube (triangulation, 0, 1, /* colorize */ true);

	triangulation.refine_global (n_refine);

	std::cout << "Number of active cells: "
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
	dof_handler.distribute_dofs (fe);

	std::cout << std::endl
			<< "Number of active global cells:   "
			<< triangulation.n_active_cells()
			<< std::endl
			<< "Number of degrees of freedom:   " << dof_handler.n_dofs()
			<< std::endl
			<< std::endl;


	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);

	/*
	 * Set up Dirichlet boundary conditions.
	 */
	const Coefficients::DirichletBC<dim> dirichlet_bc;
	for (unsigned int i = 0; i<dim; ++i)
	{
		VectorTools::interpolate_boundary_values(dof_handler,
													/*boundary id*/ 2*i, // only even boundary id
													dirichlet_bc,
													constraints);
	}

	constraints.close();


	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler,
									dsp,
									constraints,
									/*keep_constrained_dofs =*/ true); // for time stepping this is essential to be true

	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (dof_handler.n_dofs());
	system_rhs.reinit (dof_handler.n_dofs());
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

	// initialize basis iterator
	typename std::vector<DiffusionProblemBasis<dim>>::iterator
									it_basis = cell_basis_vector.begin();

	/*
	 * Integration over cells.
	 */
	for (const auto &cell: dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		cell_rhs = 0;

		cell_matrix = it_basis->get_global_element_matrix ();
		cell_rhs = it_basis->get_global_element_rhs ();

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
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
			{
				system_matrix.add(local_dof_indices[i],
							local_dof_indices[j],
							cell_matrix(i, j));
			}
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	} // end ++cell
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
	SolverControl           solver_control (1000, 1e-12);
	SolverCG<>              solver (solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);

	solver.solve (system_matrix,
				solution,
				system_rhs,
				preconditioner);

	constraints.distribute (solution);

	std::cout << "   "
			<< "(global problem)   "
			<< solver_control.last_step()
			<< " coarse CG iterations needed to obtain convergence."
			<< std::endl;
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
	// initialize basis iterator
	typename std::vector<DiffusionProblemBasis<dim>>::iterator
									it_basis = cell_basis_vector.begin ();
	for (; cell!=endc; ++cell)
	{
		// Get local
		cell->get_dof_indices (local_dof_indices);

		std::vector<double> extracted_weights (dofs_per_cell, 0);
		solution.extract_subvector_to (local_dof_indices, extracted_weights);
		it_basis->set_global_weights (extracted_weights);

		// increase syncronously
		++it_basis;
	}
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
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");
	data_out.build_patches ();

	std::ofstream output (dim == 2 ?
					"solution-ms_coarse-2d.vtu" :
					"solution-ms_coarse-3d.vtu");

	data_out.write_vtu (output);
}


/*!
 * Output function to write local multiscale solution. Only used for threading output.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::output_local_solution (
		const typename std::vector<DiffusionProblemBasis<dim>>::iterator &it_basis,
		BasisScratchData&,
		BasisCopyData&)
{
	it_basis->output_global_solution_in_cell ();
}


/*!
 * Write all local multiscale solution (threaded) and
 * a global pvtu-record.
 */
template <int dim>
void
DiffusionProblemMultiscale<dim>::output_global_fine ()
{
	// List of filenames of local outputs for master file
	std::vector<std::string> filenames_on_cell;

	// Now run them in threads
	typename std::vector<DiffusionProblemBasis<dim>>::iterator
											it_basis = cell_basis_vector.begin (),
											it_endbasis = cell_basis_vector.end ();
	WorkStream::run(it_basis,
					it_endbasis,
					*this,
					&DiffusionProblemMultiscale<dim>::output_local_solution,
					&DiffusionProblemMultiscale<dim>::fake_copy,
					BasisScratchData(),
					BasisCopyData());

	// Active cell iterator
	typename DoFHandler<dim>::active_cell_iterator
								cell = dof_handler.begin_active (),
								endc = dof_handler.end ();
	// Initialize const basis iterator again
	it_basis = cell_basis_vector.begin ();

	for (; cell!=endc; ++cell)
	{
		// Get the global file name
		filenames_on_cell.push_back ( it_basis->get_filename_global () );

		++it_basis;
	}

	// Build a*.pvtu file that points to all output files
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);

	// Names of solution components
	data_out.add_data_vector (solution, "solution");

	std::string filename_master = (dim == 2 ?
			"solution-ms_fine-2d" :
			"solution-ms_fine-3d");
	filename_master += ".pvtu";

	std::ofstream master_output (filename_master.c_str ());

	data_out.write_pvtu_record (master_output, filenames_on_cell);
}


/*!
 * @brief Run function of the object.
 *
 * Run the computation after object is built.
 */
template <int dim>
void DiffusionProblemMultiscale<dim>::run ()
{
	std::cout << std::endl
			<< "==========================================="
			<< std::endl;

	std::cout << "Solving problem in "
			<< dim << " space dimensions."
			<< std::endl;

	make_grid ();

	setup_system ();

	initialize_basis_problem ();
	compute_basis ();

	assemble_system ();

	// Now solve
	constraints.condense(system_matrix, system_rhs);
	solve_iterative ();

	send_global_weights_to_cell ();

	output_global_coarse ();
	output_global_fine ();

	std::cout << std::endl
			<< "==========================================="
			<< std::endl;
}

} // end namespace DiffusionProblem


#endif /* INCLUDE_DIFFUSION_PROBLEM_MS_HPP_ */
