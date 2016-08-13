#include <armadillo>
#include "TransverseSymmetric.h"

int main()
{
	using namespace arma;
	using namespace quantum::dmrg;
	using HamInt = TsymDMRG::HamInt;
	arma::mat hamLocal(2,2,fill::zeros);

	std::array<arma::mat,3> xxz{
		arma::mat({{0,1},{1,0}}),
		arma::mat({{0,-1},{1,0}}),
		arma::mat({{1,0},{0,-1}}),
	};
	HamInt hamInt(3);
	hamInt[0] = std::make_tuple(1.0,xxz[0]);
	hamInt[1] = std::make_tuple(-1.0,xxz[1]);
	hamInt[2] = std::make_tuple(1.0,xxz[2]);

	TsymDMRG ts(2, 100, hamLocal, hamInt);
	double eprev = 0;
	double e, ediff=10000, prevediff;
	do
	{
		prevediff = ediff;
		ts.addTwoSites();
		e = ts.getEnergy();
		ediff=(e-eprev)/2;
		eprev = e;
		std::cout << ediff << std::endl;
	}while(std::abs(ediff-prevediff) > 1e-10);
	return 0;
}
