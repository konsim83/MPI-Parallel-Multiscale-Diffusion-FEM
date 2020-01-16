/*!
 * @file  main.cxx
 * @brief Contains main function.
 * @author Konrad Simon
 * @date August 2019
 */

// My Headers
#include "base/diffusion_problem.hpp"
#include "base/diffusion_problem_ms.hpp"

using namespace dealii;


int
main(int argc, char *argv[])

{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const unsigned int n_refine = 3, n_refine_local = 7;

      const bool compute_2d = true, compute_3d = false;

      if (compute_2d)
        {
          DiffusionProblem::DiffusionProblem<2> diffusion_problem_2d_coarse(
            n_refine);
          diffusion_problem_2d_coarse.run();

          DiffusionProblem::DiffusionProblem<2> diffusion_problem_2d_fine(
            n_refine + n_refine_local);
          diffusion_problem_2d_fine.run();

          DiffusionProblem::DiffusionProblemMultiscale<2>
            diffusion_ms_problem_2d(n_refine, n_refine_local);
          diffusion_ms_problem_2d.run();
        }

      if (compute_3d)
        {
          DiffusionProblem::DiffusionProblem<3> diffusion_problem_3d_coarse(
            n_refine);
          diffusion_problem_3d_coarse.run();

          DiffusionProblem::DiffusionProblem<3> diffusion_problem_3d_fine(
            n_refine + n_refine_local);
          diffusion_problem_3d_fine.run();

          DiffusionProblem::DiffusionProblemMultiscale<3>
            diffusion_ms_problem_3d(n_refine, n_refine_local);
          diffusion_ms_problem_3d.run();
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
