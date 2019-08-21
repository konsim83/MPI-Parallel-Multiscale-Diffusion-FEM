/*!
 * @file neumann_bc.hpp
 * @brief Contains implementation of Neumann boundary conditions.
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
#include "coefficients.h"

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
	NeumannBC() : Function<dim>() {}

	virtual double value(const Point<dim> &p,
						const unsigned int component = 0) const override;
	virtual void value_list(const std::vector<Point<dim>> &points,
								std::vector<double>  &values,
								const unsigned int component = 0) const override;
};


template <int dim>
double
NeumannBC<dim>::value(const Point<dim> &p,
							   const unsigned int /*component*/) const
{
	double return_value = cos(2*PI_D*p(0)) * cos(2*PI_D*p(1));

	return return_value;
}


template <int dim>
void
NeumannBC<dim>::value_list(const std::vector<Point<dim>> &points,
								std::vector<double>  &values,
								const unsigned int /*component = 0*/) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()) );

	for ( unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = cos(2*PI_D*points[p](0)) * cos(2*PI_D*points[p](1));
	} // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_NEUMANN_BC_HPP_ */
