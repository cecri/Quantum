#include <armadillo>
#include "imps.h"
int main()
{
	using namespace arma;
	const double J = -1.0;
	const double h = -0.5;
	imps<double> ising(2, 10);
	arma::mat ham = {{J,h/2,h/2,0},{h/2,-J,0,h/2},{h/2,0,-J,h/2},{0,h/2,h/2,J}};
	double delta = 0.001;
	double E;

	arma::mat u = expmat(-ham*delta);
	for(int i = 0; i < 10000; i++)
	{
		ising.applyAB(u);
		ising.applyBA(u);
	}
	E = (ising.expectationTwoSite(ham,0) + ising.expectationTwoSite(ham,1))/2.0;
	std::cout << E << std::endl;
}
