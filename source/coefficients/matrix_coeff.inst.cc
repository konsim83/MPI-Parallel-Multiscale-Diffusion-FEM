/*!
 * @file matrix_coeff.cc
 * @brief Contains explicit instantiations of diffusion coefficient.
 * @author Konrad Simon
 * @date August 2019
 */

#include "coefficients/matrix_coeff.hpp"
#include "coefficients/matrix_coeff.tpp"

namespace Coefficients
{
  // template instantiations
  template class MatrixCoeff<2>;
  template class MatrixCoeff<3>;

} // end namespace Coefficients
