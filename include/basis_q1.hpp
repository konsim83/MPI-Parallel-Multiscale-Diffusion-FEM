/*!
 * @file basis_q1.hpp
 * @brief Contains implementation of \f$Q_1\f$-basis functions for a given quadrilateral.
 * @author Konrad Simon
 * @date August 2019
 */

#ifndef INCLUDE_BASIS_Q1_HPP_
#define INCLUDE_BASIS_Q1_HPP_

// Deal.ii
#include <deal.II/base/function.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
using namespace dealii;

/*!
 * @class BasisQ1
 * @brief Class implements scalar \f$Q_1\f$-basis functions for a given quadrilateral.
 */
template <int dim>
class BasisQ1 : public Function<dim>
{
public:
	BasisQ1 () = delete;
	BasisQ1 (const typename Triangulation<dim>::active_cell_iterator &cell);
	BasisQ1 (const BasisQ1<dim>&);

	void set_index (unsigned int index);

	virtual double value(const Point<dim> &p,
						const unsigned int component = 0) const override;
	virtual void value_list(const std::vector<Point<dim>> &points,
								std::vector<double>  &values,
								const unsigned int component = 0) const override;

private:
	/*!
	 * Index of current basis function to be evaluated.
	 */
	unsigned int 		index_basis;

	/*!
	 * Matrix columns hold coefficients of basis functions.
	 */
	FullMatrix<double>	coeff_matrix;
};



/*!
 * Copy constructor.
 *
 * @param basis
 */
template<int dim>
BasisQ1<dim>::BasisQ1 (const BasisQ1<dim>& basis)
:
Function<dim>(),
index_basis (0),
coeff_matrix (basis.coeff_matrix)
{
}


/*!
 * Constructor. Template specialization \f$dim=2\f$.
 * @param cell
 *
 * Build up coefficient matrix \f$A=(a_{ij})\f$ for basis polynomial
 * \f$\varphi_i(x,y)=a_i^0 + a_i^1 x + a_i^2 y + a_i^3 xy \f$. The \f$i\f$-th column
 * of the matrix hence contains the coefficients for the \f$i\f$-th basis associated
 * to the \f$i\f$-th vertex.
 */
template<>
BasisQ1<2>::BasisQ1 (const typename Triangulation<2>::active_cell_iterator &cell)
:
Function<2>(),
index_basis(0),
coeff_matrix(4,4)
{
	FullMatrix<double>	point_matrix(4,4);

	for (unsigned int i=0; i<4; ++i)
	{
		const Point<2>& p = cell->vertex(i);

		point_matrix(i,0) = 1;
		point_matrix(i,1) = p(0);
		point_matrix(i,2) = p(1);
		point_matrix(i,3) = p(0)*p(1);
	}

	// Columns of coeff_matrix are the coefficients of the polynomial
	coeff_matrix.invert (point_matrix);
}


/*!
 * Constructor. Template specialization \f$dim=3\f$.
 * @param cell
 *
 * Build up coefficient matrix \f$A=(a_{ij})\f$ for basis polynomial
 * \f$\varphi_i(x,y)=a_i^0 + a_i^1 x + a_i^2 y + a_i^3 xy \f$. The \f$i\f$-th column
 * of the matrix hence contains the coefficients for the \f$i\f$-th basis associated
 * to the \f$i\f$-th vertex.
 */
template<>
BasisQ1<3>::BasisQ1 (const typename Triangulation<3>::active_cell_iterator &cell)
:
Function<3>(),
index_basis(0),
coeff_matrix(8,8)
{
	FullMatrix<double>	point_matrix(8,8);

	for (unsigned int i=0; i<8; ++i)
	{
		const Point<3>& p = cell->vertex(i);

		point_matrix(i,0) = 1;
		point_matrix(i,1) = p(0);
		point_matrix(i,2) = p(1);
		point_matrix(i,3) = p(2);
		point_matrix(i,4) = p(0)*p(1);
		point_matrix(i,5) = p(1)*p(2);
		point_matrix(i,6) = p(0)*p(2);
		point_matrix(i,7) = p(0)*p(1)*p(2);
	}

	// Columns of coeff_matrix are the coefficients of the polynomial
	coeff_matrix.invert (point_matrix);
}


/*!
 * Set the index of the basis function to be evaluated.
 *
 * @param index
 */
template<int dim>
void
BasisQ1<dim>::set_index (unsigned int index)
{
	index_basis = index;
}


/*!
 * Evaluate a basis function with a preset index at one given point in 2D.
 *
 * @param p
 * @param component
 */
template<>
double
BasisQ1<2>::value (const Point<2> &p,
					const unsigned int /* component */) const
{
	double value = coeff_matrix(0,index_basis)
					+ coeff_matrix(1,index_basis)*p(0)
					+ coeff_matrix(2,index_basis)*p(1)
					+ coeff_matrix(3,index_basis)*p(0)*p(1);

	return value;
}


/*!
 * Evaluate a basis function with a preset index at one given point in 3D.
 *
 * @param p
 * @param component
 */
template<>
double
BasisQ1<3>::value (const Point<3> &p,
					const unsigned int /* component */) const
{
	double value = coeff_matrix(0,index_basis)
					+ coeff_matrix(1,index_basis)*p(0)
					+ coeff_matrix(2,index_basis)*p(1)
					+ coeff_matrix(3,index_basis)*p(2)
					+ coeff_matrix(4,index_basis)*p(0)*p(1)
					+ coeff_matrix(5,index_basis)*p(1)*p(2)
					+ coeff_matrix(6,index_basis)*p(0)*p(2)
					+ coeff_matrix(7,index_basis)*p(0)*p(1)*p(2);

	return value;
}


/*!
 * Evaluate a basis function with a preset index on a given set of points in 2D.
 *
 * @param points
 * @param values
 */
template<>
void
BasisQ1<2>::value_list(const std::vector<Point<2>> &points,
								std::vector<double>  &values,
								const unsigned int /*component = 0*/) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()) );

	for ( unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = coeff_matrix(0,index_basis)
							+ coeff_matrix(1,index_basis)*points[p](0)
							+ coeff_matrix(2,index_basis)*points[p](1)
							+ coeff_matrix(3,index_basis)*points[p](0)*points[p](1);

	} // end ++p
}


/*!
 * Evaluate a basis function with a preset index on a given set of points in 3D.
 *
 * @param points
 * @param values
 */
template<>
void
BasisQ1<3>::value_list(const std::vector<Point<3>> &points,
								std::vector<double>  &values,
								const unsigned int /*component = 0*/) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()) );

	for ( unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = coeff_matrix(0,index_basis)
							+ coeff_matrix(1,index_basis)*points[p](0)
							+ coeff_matrix(2,index_basis)*points[p](1)
							+ coeff_matrix(3,index_basis)*points[p](2)
							+ coeff_matrix(4,index_basis)*points[p](0)*points[p](1)
							+ coeff_matrix(5,index_basis)*points[p](1)*points[p](2)
							+ coeff_matrix(6,index_basis)*points[p](0)*points[p](2)
							+ coeff_matrix(7,index_basis)*points[p](0)*points[p](1)*points[p](2);

	} // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_BASIS_Q1_HPP_ */
