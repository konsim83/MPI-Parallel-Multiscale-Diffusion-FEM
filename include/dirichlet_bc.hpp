/*!
 * @file dirichlet_bc.hpp
 * @brief Contains implementation of Dirichlet boundary conditions.
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
#include "coefficients.h"

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
	DirichletBC() : Function<dim>() {}

	virtual double value(const Point<dim> &p,
						const unsigned int component = 0) const override;
	virtual void value_list(const std::vector<Point<dim>> &points,
								std::vector<double>  &values,
								const unsigned int component = 0) const override;
};


template <int dim>
double
DirichletBC<dim>::value(const Point<dim>& p,
							   const unsigned int /*component*/) const
{
	double return_value = (p(0)-0.5) * (p(0)-0.5) + (p(1)-0.5) * (p(1)-0.5);

	return return_value;
}


template <int dim>
void
DirichletBC<dim>::value_list(const std::vector<Point<dim>> &points,
								std::vector<double>  &values,
								const unsigned int /*component = 0*/) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()) );

	for ( unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = (points[p](0)-0.5) * (points[p](0)-0.5) + (points[p](1)-0.5) * (points[p](1)-0.5);
	} // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_DIRICHLET_BC_HPP_ */
