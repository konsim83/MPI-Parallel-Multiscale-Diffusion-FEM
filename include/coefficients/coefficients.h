/*!
 * @file coefficients.h
 * @brief Contains declarations and definitions global to the namespace \ref Coefficients.
 * @author Konrad Simon
 * @date August 2019
 */
#ifndef INCLUDE_COEFFICIENTS_H_
#define INCLUDE_COEFFICIENTS_H_

/*!
 * @namespace Coefficients
 * @brief Contains implementations of coefficient functions.
 */
namespace Coefficients
{
  using namespace dealii;

  /*!
   * Constant \f$\pi\f$ in double precision format.
   */
  const double PI_D = 3.14592653509793218403;

  /*!
   * Constant \f$\pi\f$ in single precision (float) format.
   */
  const float PI_F = 3.14159265358979f;

} // end namespace Coefficients

#endif /* INCLUDE_COEFFICIENTS_H_ */
