/*!
 * @file neumann_bc.tpp
 * @brief Contains implementation of Neumann boundary conditions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_NEUMANN_BC_TPP_
#define INCLUDE_NEUMANN_BC_TPP_

#include "coefficients/neumann_bc.hpp"

namespace Coefficients
{
  using namespace dealii;

  template <int dim>
  double
  NeumannBC<dim>::value(const Point<dim> &p,
                        const unsigned int /*component*/) const
  {
    double return_value = cos(2 * PI_D * p(0)) * cos(2 * PI_D * p(1));

    return return_value;
  }


  template <int dim>
  void
  NeumannBC<dim>::value_list(const std::vector<Point<dim>> &points,
                             std::vector<double> &          values,
                             const unsigned int /*component = 0*/) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p] = cos(2 * PI_D * points[p](0)) * cos(2 * PI_D * points[p](1));
      } // end ++p
  }

} // end namespace Coefficients

#endif /* INCLUDE_NEUMANN_BC_TPP_ */
