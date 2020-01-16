/*!
 * @file neumann_bc.hpp
 * @brief Contains declarations of Neumann boundary conditions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_NEUMANN_BC_HPP_
#define INCLUDE_NEUMANN_BC_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients/coefficients.h"

namespace Coefficients
{
  using namespace dealii;

  /*!
   * @class NeumannBC
   * @brief Class implements scalar Neumann conditions.
   */
  template <int dim>
  class NeumannBC : public Function<dim>
  {
  public:
    NeumannBC()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;
  };

  // exernal template instantiations
  extern template class NeumannBC<2>;
  extern template class NeumannBC<3>;

} // end namespace Coefficients

#endif /* INCLUDE_NEUMANN_BC_HPP_ */
