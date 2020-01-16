/*!
 * @file diffusion_problem.hpp
 * @brief Contains declarations of the main object.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_HPP_


/* ***************
 * Deal.ii
 * ***************
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
/* ***************
 * Deal.ii
 * ***************
 * */

// STL
#include <cmath>
#include <fstream>
#include <iostream>

// My Headers
#include "coefficients/dirichlet_bc.hpp"
#include "coefficients/matrix_coeff.hpp"
#include "coefficients/neumann_bc.hpp"
#include "coefficients/right_hand_side.hpp"
#include "config.h"

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
   * @class DiffusionProblem
   * @brief Main class to solve
   * Dirichlet-Neumann problem on a unit square.
   */
  template <int dim>
  class DiffusionProblem
  {
  public:
    /*!
     * Default constructor.
     */
    DiffusionProblem(unsigned int n_refine);

    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built. Implements theping loop.
     */
    void
    run();

  private:
    /*!
     * @brief Set up the grid with a certain number of refinements.
     *
     * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
     * numbered form \f$1,\dots,2\rm{dim}\f$.
     */
    void
    make_grid();

    /*!
     * @brief Setup sparsity pattern and system matrix.
     *
     * Compute sparsity pattern and reserve memory for the sparse system matrix
     * and a number of right-hand side vectors. Also build a constraint object
     * to take care of Dirichlet boundary conditions.
     */
    void
    setup_system();

    /*!
     * @brief Assemble the system matrix and the static right hand side.
     *
     * Assembly routine to build the time-independent (static) part.
     * Neumann boundary conditions will be put on edges/faces
     * with odd number. Constraints are not applied here yet.
     */
    void
    assemble_system();

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with AMG-preconditioning.
     */
    void
    solve_iterative();

    /*!
     * @brief Write results to disk.
     *
     * Write results to disk in vtu-format.
     */
    void
    output_results() const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    unsigned int n_refine;
  };

  // exernal template instantiations
  extern template class DiffusionProblem<2>;
  extern template class DiffusionProblem<3>;

} // end namespace DiffusionProblem


#endif /* INCLUDE_DIFFUSION_PROBLEM_HPP_ */
