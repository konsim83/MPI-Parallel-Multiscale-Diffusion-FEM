/*!
 * @file basis_q1.cc
 * @brief Contains explicit instantiations of \f$Q_1\f$-basis functions for a given quadrilateral.
 * @author Konrad Simon
 * @date August 2019
 */

#include "coefficients/basis_q1.hpp"
#include "coefficients/basis_q1.tpp"

namespace Coefficients
{
  // template instantiations
  template class BasisQ1<2>;
  template class BasisQ1<3>;

} // end namespace Coefficients
