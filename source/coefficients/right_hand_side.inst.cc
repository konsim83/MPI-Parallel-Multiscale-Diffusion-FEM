/*!
 * @file right_hand_side.cc
 * @brief Contains explicit instantiations of right-hand side.
 * @author Konrad Simon
 * @date August 2019
 */

#include "coefficients/right_hand_side.hpp"
#include "coefficients/right_hand_side.tpp"

namespace Coefficients
{
  // template instantiations
  template class RightHandSide<2>;
  template class RightHandSide<3>;

} // end namespace Coefficients
