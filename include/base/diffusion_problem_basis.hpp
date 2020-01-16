/*!
 * @file diffusion_problem_basis.hpp
 * @brief Contains declarations of multiscale basis functions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_
#define INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_

// Deal.ii
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// STL
#include <cmath>
#include <fstream>
#include <iostream>

// My Headers
#include "coefficients/basis_q1.hpp"
#include "coefficients/matrix_coeff.hpp"
#include "coefficients/neumann_bc.hpp"
#include "coefficients/right_hand_side.hpp"
#include "config.h"

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
    DiffusionProblemBasis() = delete;

    /*!
     * Default constructor.
     */
    DiffusionProblemBasis(
      unsigned int                                       n_refine_local,
      typename Triangulation<dim>::active_cell_iterator &global_cell,
      unsigned int                                       local_subdomain,
      MPI_Comm                                           mpi_communicator);

    /*!
     * Copy constructor.
     */
    DiffusionProblemBasis(const DiffusionProblemBasis<dim> &X);

    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built.
     */
    void
    run();

    /*!
     * Write out global solution in cell.
     */
    void
    output_global_solution_in_cell() const;

    /*!
     * Return the multiscale element matrix produced
     * from local basis functions.
     */
    const FullMatrix<double> &
    get_global_element_matrix() const;

    /*!
     * Get the right hand-side that was locally assembled
     * to speed up the global assembly.
     */
    const Vector<double> &
    get_global_element_rhs() const;

    /*!
     * Return filename for local pvtu record.
     */
    const std::string &
    get_filename_global();

    /*!
     * @brief Set global weights.
     * @param weights
     *
     * The coarse weights of the global solution determine
     * the local multiscale solution. They must be computed
     * and then set locally to write an output.
     */
    void
    set_global_weights(const std::vector<double> &global_weights);

    /*!
     * Set the output flag to write basis functions to disk as vtu.
     * @param flag
     */
    void
    set_output_flag(bool flag);

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
     * @brief Assemble the gloabl element matrix and the gobla right hand side.
     */
    void
    assemble_global_element_matrix();

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with SSOR-preconditioning.
     */
    void
    solve_iterative(unsigned int index_basis);

    /*!
     * @brief Write basis results to disk.
     *
     * Write basis results to disk in vtu-format.
     */
    void
    output_basis() const;

    /*!
     * Define the gloabl filename for pvtu-file in global output.
     */
    void
    set_filename_global();

    MPI_Comm mpi_communicator;

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    std::vector<AffineConstraints<double>> constraints_vector;
    std::vector<Point<dim>>                corner_points;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> diffusion_matrix;
    SparseMatrix<double> system_matrix;

    std::string filename_global;

    /*!
     * Solution vector.
     */
    std::vector<Vector<double>> solution_vector;

    /*!
     * Contains the right-hand side.
     */
    Vector<double>
      global_rhs; // this is only for the global assembly (speed-up)

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system..
     */
    Vector<double> system_rhs;

    /*!
     * Holds global multiscale element matrix.
     */
    FullMatrix<double> global_element_matrix;
    bool               is_built_global_element_matrix;

    /*!
     * Holds global multiscale element right-hand side.
     */
    Vector<double> global_element_rhs;

    /*!
     * Weights of multiscale basis functions.
     */
    std::vector<double> global_weights;
    bool                is_set_global_weights;

    /*!
     * Global solution
     */
    Vector<double> global_solution;

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

  // exernal template instantiations
  extern template class DiffusionProblemBasis<2>;
  extern template class DiffusionProblemBasis<3>;

} // end namespace DiffusionProblem

#endif /* INCLUDE_DIFFUSION_PROBLEM_BASIS_HPP_ */
