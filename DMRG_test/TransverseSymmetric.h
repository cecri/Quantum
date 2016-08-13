#ifndef CY_QUANTUM_DMRG_TRANSVERSE_SYMMETRIC_H
#define CY_QUANTUM_DMRG_TRANSVERSE_SYMMETRIC_H
#include <armadillo>
namespace quantum
{
namespace dmrg
{
class TsymDMRG
{
public:
	using HamInt = std::vector<std::tuple<double, arma::mat> >;
private:
	std::size_t d_;
	std::size_t D_;
	std::size_t bondDim_;
	std::size_t length_;

	double curEnergy_;
	
	HamInt hamInt_;
	arma::mat hamLocal_;

	std::vector<arma::mat> hamL_; //hamL_ contains the interaction Hamiltonian in left
	arma::mat hamLeft_; //in each step, hamLeft_ is updated using the previous basis set and the added site.
	arma::vec svs_;

	void calcReducedBasis();

public:
	/* D: bond dimension
	 * ham: sparse hamiltonian for 4 site
	 * */
	TsymDMRG(std::size_t d, std::size_t bondDim, const arma::mat& hamLocal, const HamInt& hamInt);
	~TsymDMRG() = default;
	TsymDMRG(const TsymDMRG&) = default;
	TsymDMRG(TsymDMRG&&) = default;

	void addTwoSites();

	void test(const arma::vec& v);

	double getEnergy()
	{
		return curEnergy_;
	}
	int getLength()
	{
		return length_;
	}
};
}
}
#endif
