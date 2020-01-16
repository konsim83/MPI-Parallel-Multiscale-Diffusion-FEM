/*!
 * @file matrix_coeff.tpp
 * @brief Contains implementation of diffusion coefficient.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_MATRIX_COEFF_TPP_
#define INCLUDE_MATRIX_COEFF_TPP_

#include "matrix_coeff.hpp"

namespace Coefficients
{
  using namespace dealii;

  template <>
  MatrixCoeff<2>::MatrixCoeff()
    : TensorFunction<2, 2>()
  {
    rot[0][0] = cos(alpha);
    rot[0][1] = sin(alpha);
    rot[1][0] = -sin(alpha);
    rot[1][1] = cos(alpha);
  }


  template <>
  MatrixCoeff<3>::MatrixCoeff()
    : TensorFunction<2, 3>()
  {
    rot[0][0] = cos(alpha) * cos(gamma) - sin(alpha) * cos(beta) * sin(gamma);
    rot[0][1] = -cos(alpha) * sin(gamma) - sin(alpha) * cos(beta) * cos(gamma);
    rot[0][2] = sin(alpha) * sin(beta);
    rot[1][0] = sin(alpha) * cos(gamma) + cos(alpha) * cos(beta) * sin(gamma);
    rot[1][1] = -sin(alpha) * sin(gamma) + cos(alpha) * cos(beta) * cos(gamma);
    rot[1][2] = -cos(alpha) * sin(beta);
    rot[2][0] = sin(beta) * sin(gamma);
    rot[2][1] = sin(beta) * cos(gamma);
    rot[2][2] = cos(beta);
  }


  template <int dim>
  Tensor<2, dim>
  MatrixCoeff<dim>::value(const Point<dim> &p) const
  {
    Tensor<2, dim> value;
    value.clear();

    for (unsigned int d = 0; d < dim; ++d)
      {
        value[d][d] =
          1.0 * (1.0 - scale_factor *
                         (0.5 * sin(2 * PI_D * k * p(0)) +
                          0.5 * sin(2 * PI_D * k *
                                    p(1)))); /* Must be positive definite. */
      }

    value = rot * value * transpose(rot);

    return value;
  }


  template <int dim>
  void
  MatrixCoeff<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<Tensor<2, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p].clear();

        for (unsigned int d = 0; d < dim; ++d)
          {
            values[p][d][d] =
              1.0 *
              (1.0 -
               scale_factor *
                 (0.5 * sin(2 * PI_D * k * points[p](0)) +
                  0.5 * sin(2 * PI_D * k *
                            points[p](1)))); /* Must be positive definite. */
          }

        values[p] = rot * values[p] * transpose(rot);
      }
  }

} // end namespace Coefficients

#endif /* INCLUDE_MATRIX_COEFF_TPP_ */
