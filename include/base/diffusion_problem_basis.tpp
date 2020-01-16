/*!
 * @file diffusion_problem_basis.tpp
 * @brief Contains implementation of multiscale basis functions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIFFUSION_PROBLEM_BASIS_TPP_
#define INCLUDE_DIFFUSION_PROBLEM_BASIS_TPP_

#include "base/diffusion_problem_basis.hpp"

namespace DiffusionProblem
{
  using namespace dealii;


  template <int dim>
  DiffusionProblemBasis<dim>::DiffusionProblemBasis(
    unsigned int                                       n_refine_local,
    typename Triangulation<dim>::active_cell_iterator &global_cell,
    unsigned int                                       local_subdomain,
    MPI_Comm                                           mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , fe(1)
    , dof_handler(triangulation)
    , constraints_vector(GeometryInfo<dim>::vertices_per_cell)
    , corner_points(GeometryInfo<dim>::vertices_per_cell)
    , filename_global("")
    , solution_vector(GeometryInfo<dim>::vertices_per_cell)
    , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
    , is_built_global_element_matrix(false)
    , global_element_rhs(fe.dofs_per_cell)
    , global_weights(fe.dofs_per_cell, 0)
    , is_set_global_weights(false)
    , n_refine_local(n_refine_local)
    , global_cell_id(global_cell->id())
    , local_subdomain(local_subdomain)
    , basis_q1(global_cell)
    , output_flag(false)
    , verbose(true)
  {
    // set corner points
    for (unsigned int vertex_n = 0;
         vertex_n < GeometryInfo<dim>::vertices_per_cell;
         ++vertex_n)
      {
        corner_points[vertex_n] = global_cell->vertex(vertex_n);
      }
  }


  template <int dim>
  DiffusionProblemBasis<dim>::DiffusionProblemBasis(
    const DiffusionProblemBasis<dim> &X)
    : // triangulation(X.triangulation), // only possible if object is empty
    mpi_communicator(X.mpi_communicator)
    , fe(X.fe)
    , dof_handler(triangulation)
    , // must be constructed deliberately
    constraints_vector(X.constraints_vector)
    , corner_points(X.corner_points)
    , sparsity_pattern(X.sparsity_pattern)
    , // only possible if object is empty
    diffusion_matrix(X.diffusion_matrix)
    , // only possible if object is empty
    system_matrix(X.system_matrix)
    , // only possible if object is empty
    filename_global(X.filename_global)
    , solution_vector(X.solution_vector)
    , global_rhs(X.global_rhs)
    , system_rhs(X.system_rhs)
    , global_element_matrix(X.global_element_matrix)
    , is_built_global_element_matrix(X.is_built_global_element_matrix)
    , global_element_rhs(X.global_element_rhs)
    , global_weights(X.global_weights)
    , is_set_global_weights(X.is_set_global_weights)
    , global_solution(X.global_solution)
    , n_refine_local(X.n_refine_local)
    , global_cell_id(X.global_cell_id)
    , local_subdomain(X.local_subdomain)
    , basis_q1(X.basis_q1)
    , output_flag(X.output_flag)
    , verbose(X.verbose)
  {
    //	triangulation.copy_triangulation (X.triangulation);
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::make_grid()
  {
    GridGenerator::general_cell(triangulation,
                                corner_points,
                                /* colorize faces */ false);

    triangulation.refine_global(n_refine_local);
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    if (verbose)
      std::cout << "Global cell id  " << global_cell_id.to_string()
		<< " (subdomain = " << local_subdomain << "):   "
                << triangulation.n_active_cells() << " active fine cells --- "
                << dof_handler.n_dofs() << " subgrid dof" << std::endl;

    /*
     * Set up Dirichlet boundary conditions and sparsity pattern.
     */
    DynamicSparsityPattern dsp(dof_handler.n_dofs());

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<dim>::vertices_per_cell;
         ++index_basis)
      {
        basis_q1.set_index(index_basis);

        constraints_vector[index_basis].clear();
        DoFTools::make_hanging_node_constraints(
          dof_handler, constraints_vector[index_basis]);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          /*boundary id*/ 0,
          basis_q1,
          constraints_vector[index_basis]);
        constraints_vector[index_basis].close();
      }

    DoFTools::make_sparsity_pattern(
      dof_handler,
      dsp,
      constraints_vector[0], // sparsity pattern is the same for each basis
      /*keep_constrained_dofs =*/true); // for time stepping this is essential
                                        // to be true
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    diffusion_matrix.reinit(sparsity_pattern);

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<dim>::vertices_per_cell;
         ++index_basis)
      {
        solution_vector[index_basis].reinit(dof_handler.n_dofs());
      }
    system_rhs.reinit(dof_handler.n_dofs());
    global_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_diffusion_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    /*
     * Matrix coefficient and vector to store the values.
     */
    const Coefficients::MatrixCoeff<dim> matrix_coeff;
    std::vector<Tensor<2, dim>>          matrix_coeff_values(n_q_points);

    /*
     * Right hand side and vector to store the values.
     */
    const Coefficients::RightHandSide<dim> right_hand_side;
    std::vector<double>                    rhs_values(n_q_points);

    /*
     * Integration over cells.
     */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_diffusion_matrix = 0;
        cell_rhs              = 0;

        fe_values.reinit(cell);

        // Now actually fill with values.
        matrix_coeff.value_list(fe_values.get_quadrature_points(),
                                matrix_coeff_values);
        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_diffusion_matrix(i, j) +=
                      fe_values.shape_grad(i, q_index) *
                      matrix_coeff_values[q_index] *
                      fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index);
                  } // end ++j

                cell_rhs(i) += fe_values.shape_value(i, q_index) *
                               rhs_values[q_index] * fe_values.JxW(q_index);
              } // end ++i
          }     // end ++q_index

        // get global indices
        cell->get_dof_indices(local_dof_indices);
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


  template <int dim>
  void
  DiffusionProblemBasis<dim>::assemble_global_element_matrix()
  {
    // First, reset.
    global_element_matrix = 0;

    // Get lengths of tmp vectors for assembly
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> tmp(dof_handler.n_dofs());

    // This assembles the local contribution to the global global matrix
    // with an algebraic trick. It uses the local system matrix stored in
    // the respective basis object.
    for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test)
      {
        // set an alias name
        const Vector<double> &test_vec = solution_vector[i_test];

        for (unsigned int i_trial = 0; i_trial < dofs_per_cell; ++i_trial)
          {
            // set an alias name
            const Vector<double> &trial_vec = solution_vector[i_trial];

            // tmp = system_matrix*trial_vec
            diffusion_matrix.vmult(tmp, trial_vec);

            // global_element_diffusion_matrix = test_vec*tmp
            global_element_matrix(i_test, i_trial) += (test_vec * tmp);

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
  void
  DiffusionProblemBasis<dim>::solve_iterative(unsigned int index_basis)
  {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>    solver(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.6);

    solver.solve(system_matrix,
                 solution_vector[index_basis],
                 system_rhs,
                 preconditioner);

    constraints_vector[index_basis].distribute(solution_vector[index_basis]);

    if (verbose)
      std::cout << "   "
                << "(cell   " << global_cell_id.to_string() << ") "
                << "(basis   " << index_basis << ")   "
                << solver_control.last_step()
                << " fine CG iterations needed to obtain convergence."
                << std::endl;
  }


  template <int dim>
  const FullMatrix<double> &
  DiffusionProblemBasis<dim>::get_global_element_matrix() const
  {
    return global_element_matrix;
  }


  template <int dim>
  const Vector<double> &
  DiffusionProblemBasis<dim>::get_global_element_rhs() const
  {
    return global_element_rhs;
  }


  template <int dim>
  const std::string &
  DiffusionProblemBasis<dim>::get_filename_global()
  {
    return filename_global;
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::set_output_flag(bool flag)
  {
    output_flag = flag;
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::set_global_weights(
    const std::vector<double> &weights)
  {
    // Copy assignment of global weights
    global_weights = weights;

    // reinitialize the global solution on this cell
    global_solution.reinit(dof_handler.n_dofs());

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // Set global solution using the weights and the local basis.
    for (unsigned int index_basis = 0; index_basis < dofs_per_cell;
         ++index_basis)
      {
        // global_solution = 1*global_solution +
        // global_weights[index_basis]*solution_vector[index_basis]
        global_solution.sadd(1,
                             global_weights[index_basis],
                             solution_vector[index_basis]);
      }

    is_set_global_weights = true;
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::set_filename_global()
  {
    filename_global +=
      (dim == 2 ? "solution-ms_fine-2d" : "solution-ms_fine-3d");

    filename_global += "." + Utilities::int_to_string(local_subdomain, 5);
    filename_global += ".cell-" + global_cell_id.to_string() + ".vtu";
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::output_basis() const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<dim>::vertices_per_cell;
         ++index_basis)
      {
        data_out.add_data_vector(solution_vector[index_basis],
                                 "basis_" +
                                   Utilities::int_to_string(index_basis, 1));
      }
    data_out.build_patches();

    std::string filename = "basis";
    filename +=
      "." +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 5);
    filename += ".cell-" + global_cell_id.to_string();
    filename += ".vtu";

    std::ofstream output(dim == 2 ? "2d-" + filename : "3d-" + filename);

    data_out.write_vtu(output);
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::output_global_solution_in_cell() const
  {
    Assert(is_set_global_weights,
           ExcMessage("Global weights must be set first."));

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(global_solution, "solution");
    data_out.build_patches();

    std::ofstream output(filename_global.c_str());
    data_out.write_vtu(output);
  }


  template <int dim>
  void
  DiffusionProblemBasis<dim>::run()
  {
    make_grid();

    setup_system();

    assemble_system();

    set_filename_global();

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<dim>::vertices_per_cell;
         ++index_basis)
      {
        // reset everything
        system_rhs.reinit(solution_vector[index_basis].size());
        system_matrix.reinit(sparsity_pattern);

        system_matrix.copy_from(diffusion_matrix);

        // Now take care of constraints
        constraints_vector[index_basis].condense(system_matrix, system_rhs);

        // Now solve
        solve_iterative(index_basis);
      }

    assemble_global_element_matrix();

    if (output_flag)
      output_basis();

    if (verbose)
      std::cout << std::endl;
  }

} // end namespace DiffusionProblem

#endif /* INCLUDE_DIFFUSION_PROBLEM_BASIS_TPP_ */
