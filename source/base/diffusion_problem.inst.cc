/*!
 * @file diffusion_problem.cc
 * @brief Contains explicit instantiations of the main object.
 * @author Konrad Simon
 * @date August 2019
 */


#include "base/diffusion_problem.hpp"
#include "base/diffusion_problem.tpp"

namespace DiffusionProblem
{
  // template instantiations
  template class DiffusionProblem<2>;
  template class DiffusionProblem<3>;

} // end namespace DiffusionProblem
