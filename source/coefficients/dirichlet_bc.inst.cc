/*!
 * @file dirichlet_bc.cc
 * @brief Contains explicit instantiations of Dirichlet boundary conditions.
 * @author Konrad Simon
 * @date August 2019
 */

#include "coefficients/dirichlet_bc.hpp"
#include "coefficients/dirichlet_bc.tpp"

namespace Coefficients
{
  // template instantiations
  template class DirichletBC<2>;
  template class DirichletBC<3>;

} // end namespace Coefficients
