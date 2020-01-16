/*!
 * @file dirichlet_bc.hpp
 * @brief Contains declarations of Dirichlet boundary conditions.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_DIRICHLET_BC_HPP_
#define INCLUDE_DIRICHLET_BC_HPP_

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
   * @class DirichletBC
   * @brief Class implements scalar Dirichlet conditions.
   */
  template <int dim>
  class DirichletBC : public Function<dim>
  {
  public:
    DirichletBC()
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
  extern template class DirichletBC<2>;
  extern template class DirichletBC<3>;

} // end namespace Coefficients

#endif /* INCLUDE_DIRICHLET_BC_HPP_ */
