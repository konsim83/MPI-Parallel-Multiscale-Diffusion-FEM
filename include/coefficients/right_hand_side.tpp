/*!
 * @file right_hand_side.tpp
 * @brief Contains implementation of right-hand side.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_RIGHT_HAND_SIDE_TPP_
#define INCLUDE_RIGHT_HAND_SIDE_TPP_

#include "coefficients/right_hand_side.hpp"

namespace Coefficients
{
  using namespace dealii;

  template <int dim>
  double
  RightHandSide<dim>::value(const Point<dim> & /*p*/,
                            const unsigned int /*component*/) const
  {
    double return_value = 2.0;

    return return_value;
  }

  template <int dim>
  void
  RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double> &          values,
                                 const unsigned int /*component = 0*/) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p] = 2.0;
      } // end ++p
  }

} // end namespace Coefficients

#endif /* INCLUDE_RIGHT_HAND_SIDE_TPP_ */
