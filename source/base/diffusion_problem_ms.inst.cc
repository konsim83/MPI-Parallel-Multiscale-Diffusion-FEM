/*!
 * @file diffusion_problem_ms.cc
 * @brief Contains explicit instantiations of the main object for multiscale FEM.
 * @author Konrad Simon
 * @date August 2019
 */


#include "base/diffusion_problem_ms.hpp"
#include "base/diffusion_problem_ms.tpp"

namespace DiffusionProblem
{
  // template instantiations
  template class DiffusionProblemMultiscale<2>;
  template class DiffusionProblemMultiscale<3>;

} // end namespace DiffusionProblem
