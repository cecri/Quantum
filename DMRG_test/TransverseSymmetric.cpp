#include <cassert>
#include <algorithm>
#include "TransverseSymmetric.h"


namespace QD = quantum::dmrg;
namespace {

//vo = H*v
inline void productHamVec(std::size_t d, std::size_t D, const arma::mat& hamLeft, const quantum::dmrg::TsymDMRG::HamInt& hamInt, const std::vector<arma::mat>& hamL, const arma::mat& hamLocal, const arma::vec& v, arma::vec& vo)
{
	vo.zeros();
	//hamLeft
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		for(std::size_t j = 0; j < D; j++)
		{
			vo.at(i) += v.at(j+(i/D)*D)*hamLeft.at(i%D,j);
		}
	}
	//hamRight
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		int np = (i/(D*d))%D;
		int ip = i - np*D*d;
		for(std::size_t j = 0; j < D; j++)
		{
			vo.at(i) += v.at(ip+j*D*d)*hamLeft.at(np,j);
		}
	}

	//hamLO
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		for(std::size_t k = 0; k < hamInt.size(); k++)
		{
			const double p = std::get<0>(hamInt[k]);
			const arma::mat& w = std::get<1>(hamInt[k]);
			for(std::size_t j = 0; j < D*d; j++)
			{
				vo.at(i) += p*v.at(j+(i/(D*d))*D*d)*hamL[k].at(j%D,i%D)*w.at(j/D,(i%(D*d))/D);
			}
		}
	}

	//hamRO
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		for(std::size_t k = 0; k < hamInt.size(); k++)
		{
			const double p = std::get<0>(hamInt[k]);
			const arma::mat& w = std::get<1>(hamInt[k]);

			for(std::size_t j = 0; j < D*d; j++)
			{
				vo.at(i) += p*v.at(i%(D*d)+j*D*d)*hamL[k].at(j%D,(i/(D*d))%D)*w.at(j/D,i/(D*d*D));
			}
		}
	}
	//hamLocals
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		const int n1 = (i/D)%d;
		const int ip = i - n1*D;
		for(std::size_t k = 0; k < d; k++)
		{
			vo.at(i) += v.at(ip+k*D)*hamLocal.at(k,n1);
		}
	}
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		const int n2 = (i/(D*d*D));
		for(std::size_t k = 0; k < d; k++)
		{
			vo.at(i) += v.at(i%(D*d*D)+k*(D*d*D))*hamLocal.at(k,n2);
		}
	}
	//hamOO
#pragma omp parallel for schedule(static, 4)
	for(std::size_t i = 0; i < D*d*D*d; i++)
	{
		const int n1 = (i/D)%d;
		const int n2 = (i/(D*d*D));
		const int ip = i - n1*D - n2*(D*d*D);
		for(std::size_t k = 0; k < hamInt.size(); k++)
		{
			double p = std::get<0>(hamInt[k]);
			const arma::mat& w = std::get<1>(hamInt[k]);
			for(std::size_t j = 0; j < d*d; j++)
			{
				vo.at(i) += p*v.at(ip+(j%d)*D+(j/d)*D*d*D)*w.at(j%d,n1)*w.at(j/d,n2);
			}
		}
	}
}
}//namespace


QD::TsymDMRG::TsymDMRG(std::size_t d, std::size_t bondDim, const arma::mat& hamLocal, const HamInt& hamInt)
	: d_(d), D_(d), bondDim_(bondDim), length_(2), hamInt_(hamInt), hamLocal_(hamLocal), hamLeft_(hamLocal)
{
	using namespace arma;
	using std::abs;
	
	hamL_.resize(hamInt.size());
	std::transform(hamInt.begin(), hamInt.end(), hamL_.begin(), [](const std::tuple<double,arma::mat>& t){
		return std::get<1>(t);
	});
}

void QD::TsymDMRG::calcReducedBasis()
{	
	using std::sqrt;
	using std::abs;
	using namespace arma;

	vec gs;
	const std::size_t dim = D_*d_*D_*d_;
	vec v[2];
	v[0].resize(dim);
	v[0].randu();
	v[0] /= norm(v[0]);
	v[1].resize(dim);

	double eprev;
	double e = std::numeric_limits<double>::max();


	static const double tol = 1e-10;
	vec w, ww;
	w.resize(dim);
	ww.resize(dim);
	do{
		eprev = e;
		productHamVec(d_,D_,hamLeft_,hamInt_,hamL_,hamLocal_,v[0],w);
		double hex = dot(v[0],w);
		v[1] = w;
		v[1] -= hex*v[0];
		double n = norm(v[1]);
		v[1]/=n;

		//construct reduced 2*2 mat
		double a, d;
		double b;

		productHamVec(d_,D_,hamLeft_,hamInt_,hamL_,hamLocal_,v[1],ww);
		a = hex;
		b = dot(v[1],w);
		d = dot(v[1],ww);
		e = (a+d)/2.0 - sqrt((a-d)*(a-d)+4*b*b)/2.0;
		v[0] *= b;
		v[0] += (e-a)*v[1];
		v[0] = normalise(v[0]);
	}while(abs(eprev/e-1.0) > tol);

	gs = v[0];
	curEnergy_ = e;
	mat coeff(D_*d_,D_*d_,fill::zeros);

	for(size_t k = 0; k < D_*d_; k++)
	{
		for(size_t j = 0; j < D_*d_; j++)
		{
			for(size_t i = 0; i < D_*d_; i++)
			{
				coeff(i,j) += gs.at(i+k*(D_*d_))*gs.at(j+k*(D_*d_));
			}
		}
	}

	vec evals;
	mat evecs;
	eig_sym(evals,evecs,coeff);
	size_t Dnew = std::min<size_t>(d_*D_,bondDim_);

	svs_ = evals.tail(Dnew);

	arma::mat hamLeft(Dnew,Dnew, fill::zeros);

#pragma omp parallel for schedule(static, 4)
	for(size_t i = 0; i < Dnew; i++)
	{
		for(size_t j = 0; j < Dnew; j++)
		{
			for(size_t m = 0; m < d_; m++)
			{
				for(size_t l = 0; l < D_; l++)
				{
					for(size_t k = 0; k < D_; k++)
					{
						hamLeft.at(Dnew-1-i,Dnew-1-j) += 
							evecs.at(k+m*D_,D_*d_-1-i)*evecs.at(l+m*D_,D_*d_-1-j)*hamLeft_.at(k,l);
					}
				}
			}
		}
	}

#pragma omp parallel for schedule(static, 4)
	for(size_t i = 0; i < Dnew; i++)
	{
		for(size_t j = 0; j < Dnew; j++)
		{
			for(size_t a = 0; a < hamInt_.size(); a++)
			{
				const double p = std::get<0>(hamInt_[a]);
				for(size_t k = 0; k < D_*d_; k++)
				{
					for(size_t l = 0; l < D_*d_; l++)
					{
						hamLeft.at(Dnew-1-i,Dnew-1-j) += p*evecs.at(k,D_*d_-1-i)*evecs.at(l,D_*d_-1-j)
							*hamL_[a].at(k%D_,l%D_)*std::get<1>(hamInt_[a]).at(k/D_,l/D_);
					}
				}
			}
		}
	}

#pragma omp parallel for schedule(static, 4)
	for(size_t i = 0; i < Dnew; i++)
	{
		for(size_t j = 0; j < Dnew; j++)
		{
			for(size_t k = 0; k < d_; k++)
			{
				for(size_t l = 0; l < d_; l++)
				{
					for(size_t m = 0; m < D_; m++)
					{
						hamLeft.at(Dnew-1-i,Dnew-1-j) += evecs.at(k*D_+m,D_*d_-1-i)*evecs.at(l*D_+m,D_*d_-1-j)
							*hamLocal_.at(k,l);
					}
				}
			}
		}
	}


	std::vector<arma::mat> hamL(hamInt_.size(), arma::mat(Dnew, Dnew, fill::zeros));
	for(size_t a = 0; a < hamInt_.size(); a++)
	{
#pragma omp parallel for schedule(static, 4)
		for(size_t i = 0; i < Dnew; i++)
		{
			for(size_t j = 0; j < Dnew; j++)
			{
				for(size_t l = 0; l < d_*d_; l++)
				{
					for(size_t k = 0; k < D_; k++)
					{
						hamL[a].at(Dnew-1-i,Dnew-1-j) += 
							evecs.at(k+(l/d_)*D_,D_*d_-1-i)*evecs.at(k+(l%d_)*D_,D_*d_-1-j)*std::get<1>(hamInt_[a]).at(l/d_,l%d_);
					}
				}
			}
		}
	}
	D_ = Dnew;
	hamLeft_ = std::move(hamLeft);
	hamL_ = std::move(hamL);
}

void QD::TsymDMRG::addTwoSites()
{
	using namespace arma;
	calcReducedBasis();
	length_ += 2;
}
