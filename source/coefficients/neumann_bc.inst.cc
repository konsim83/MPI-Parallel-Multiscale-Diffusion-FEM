/*!
 * @file neumann_bc.cc
 * @brief Contains explicit instantiations of Neumann boundary conditions.
 * @author Konrad Simon
 * @date August 2019
 */

#include "coefficients/neumann_bc.hpp"
#include "coefficients/neumann_bc.tpp"

namespace Coefficients
{
  // template instantiations
  template class NeumannBC<2>;
  template class NeumannBC<3>;

} // end namespace Coefficients
