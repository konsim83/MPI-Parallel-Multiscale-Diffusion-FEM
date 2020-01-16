/*!
 * @file diffusion_problem_ms.hpp
 * @brief Contains declarations of the main object for multiscale FEM.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_MS_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_MS_HPP_

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
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

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
#include <map>
#include <vector>

// My Headers
#include "base/diffusion_problem_basis.hpp"
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
   * @class DiffusionProblemMultiscale
   * @brief Main class to solve
   * Dirichlet-Neumann problem on a unit square with
   * multiscale FEM.
   */
  template <int dim>
  class DiffusionProblemMultiscale
  {
  public:
    /*!
     * Constructor.
     */
    DiffusionProblemMultiscale(unsigned int n_refine,
                               unsigned int n_refine_local);

    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built.
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
     * Set all relevant data to local basis object and initialize the basis
     * fully. Then compute.
     */
    void
    initialize_and_compute_basis();

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
     * @brief Send coarse weights to corresponding local cell.
     *
     * After the coarse (global) weights have been computed they
     * must be set to the local basis object and stored there.
     * This is necessary to write the local multiscale solution.
     */
    void
    send_global_weights_to_cell();

    /*!
     * @brief Write coarse solution to disk.
     *
     * Write results for coarse solution to disk in vtu-format.
     */
    void
    output_global_coarse() const;

    /*!
     * Write all local multiscale solution (threaded) and
     * a global pvtu-record.
     */
    void
    output_global_fine();

    /*!
     * Collect local file names on all mpi processes to write
     * the global pvtu-record.
     */
    std::vector<std::string>
    collect_filenames_on_mpi_process();

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    /*!
     *
     */
    LA::MPI::SparseMatrix system_matrix;

    /*!
     * Solution vector containing weights at the dofs.
     */
    LA::MPI::Vector locally_relevant_solution;

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system.
     */
    LA::MPI::Vector system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

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
    std::map<CellId, DiffusionProblemBasis<dim>> cell_basis_map;
  };

  // exernal template instantiations
  extern template class DiffusionProblemMultiscale<2>;
  extern template class DiffusionProblemMultiscale<3>;

} // end namespace DiffusionProblem


#endif /* INCLUDE_DIFFUSION_PROBLEM_MS_HPP_ */
