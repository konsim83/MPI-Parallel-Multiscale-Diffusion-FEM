// This program tests the functionality of my_vector_tools (parallel
// and serial projection on FE spaces).


// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

// C++ STL
#include <iostream>

using namespace dealii;


///////////////////////////////////
///////////////////////////////////
int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);

  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                            0));

  try
    {
      pcout << "Dummy test successful on "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " MPI ranks."
            << std::endl;
    }
  catch (...)
    {
      pcout << "Dummy test failed on "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " MPI ranks."
            << std::endl;
    }
}
///////////////////////////////////
///////////////////////////////////
