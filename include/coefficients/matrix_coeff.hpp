/*!
 * @file matrix_coeff.hpp
 * @brief Contains declarations of diffusion coefficient.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_MATRIX_COEFF_HPP_
#define INCLUDE_MATRIX_COEFF_HPP_

// Deal.ii
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
  using namespace dealii;

  /*!
   * @class MatrixCoeff
   * @brief Diffusion coefficient.
   *
   * Class implements a matrix valued diffusion coefficient.
   * This coefficient must be positive definite.
   */
  template <int dim>
  class MatrixCoeff : public TensorFunction<2, dim>
  {
  public:
    MatrixCoeff();

    virtual Tensor<2, dim>
    value(const Point<dim> &point) const override;
    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<2, dim>> &  values) const override;

  private:
    const int    k            = 57;
    const double scale_factor = 0.9999999;

    const double   alpha = PI_D / 3, beta = PI_D / 6, gamma = PI_D / 4;
    Tensor<2, dim> rot;
  };

  // exernal template instantiations
  extern template class MatrixCoeff<2>;
  extern template class MatrixCoeff<3>;

} // end namespace Coefficients

#endif /* INCLUDE_MATRIX_COEFF_HPP_ */
