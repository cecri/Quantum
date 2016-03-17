#ifndef CY_IMPS_H
#define CY_IMPS_H
#include <armadillo>
#include <array>
#include "RandomReal.h"

template<typename T>
class imps
{
private:
	int d_;
	int chi_;
	std::array<arma::Mat<T>,2> gamma_;
	std::array<arma::vec,2> lambda_;
public:
	imps() = delete;
	~imps() = default;
	imps(const imps&) = default;
	imps& operator=(const imps&) = default;

	imps(int d, int chi);

	inline int idx(int i1, int i2)
	{
		return i1*d_+i2;
	}
	//apply AB for n = 0, BA for n = 1
	void applyTwoSite(const arma::Mat<T>& u, int n);
	T expectationTwoSite(const arma::Mat<T>& h, int n);

	void applyAB(const arma::Mat<T>& u){ applyTwoSite(u,0);}
	void applyBA(const arma::Mat<T>& u){ applyTwoSite(u,1);}

	T expectationAB(const arma::Mat<T>& h){ return expectationTwoSite(h,0); }
	T expectationBA(const arma::Mat<T>& h){ return expectationTwoSite(h,1); }


	arma::Mat<T> getGammaA()
	{
		return gamma_[0];
	}
	arma::Mat<T> getGammaB()
	{
		return gamma_[1];
	}
	arma::vec getLambdaA()
	{
		return lambda_[0];
	}
	arma::vec getLambdaB()
	{
		return lambda_[1];
	}

	arma::Col<T> applyTMLeft(arma::Col<T>& v);
	arma::Row<T> applyTMRight(arma::Row<T>& v);
};

#include "imps.cpp"
#endif
