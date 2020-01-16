/*!
 * @file diffusion_problem_basis.cc
 * @brief Contains explicit instantiations of multiscale basis functions.
 * @author Konrad Simon
 * @date August 2019
 */


#include "base/diffusion_problem_basis.hpp"
#include "base/diffusion_problem_basis.tpp"

namespace DiffusionProblem
{
  // template instantiations
  template class DiffusionProblemBasis<2>;
  template class DiffusionProblemBasis<3>;

} // end namespace DiffusionProblem
