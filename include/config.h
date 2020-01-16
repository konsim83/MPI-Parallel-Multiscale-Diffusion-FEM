#ifndef INCLUDE_CONFIG_H_
#define INCLUDE_CONFIG_H_

#include <deal.II/lac/generic_linear_algebra.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <type_traits>


#define FORCE_USE_OF_TRILINOS

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

/*!
 * Numeric epsilon for types::REAL. Interface to C++ STL.
 */
static const double double_eps = std::numeric_limits<double>::epsilon();

/*!
 * Numeric minimum for types::REAL. Interface to C++ STL.
 */
static const double double_min = std::numeric_limits<double>::min();

/*!
 * Numeric maximum for types::REAL. Interface to C++ STL.
 */
static const double double_max = std::numeric_limits<double>::max();

/*!
 * Function to compare two non-integer values. Return a bool. Interface to C++
 * STL.
 */
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
is_approx(T x, T y, int ulp = 2)
{
  /* Machine epsilon has to be scaled to the magnitude
   * of the values used and multiplied by the desired precision
   * in ULPs (units in the last place) */
  return std::abs(x - y) <=
           std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         /* unless the result is subnormal. */
         || std::abs(x - y) < std::numeric_limits<T>::min();
}


#endif /* INCLUDE_CONFIG_H_ */
