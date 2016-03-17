#include "imps.h"

constexpr std::complex<double> I(0.0,1.0);
using namespace arma;

namespace 
{
	inline std::complex<double> bar(std::complex<double> c)
	{
		return conj(c);
	}
	inline double bar(double c)
	{
		return c;
	}
	inline Mat<double> bar(const Mat<double>& mat)
	{
		return mat;
	}
	inline Mat<cx_double> bar(const Mat<cx_double>& mat)
	{
		return conj(mat);
	}

	template<typename T>
	void reduceDim(int d, int D, const Mat<T>& theta, Mat<T>& gamLeft, Mat<T>& gamRight, vec& lamIn, const vec& lamOut)
	{
		Mat<T> U, V;
		vec s;
		svd(U,s,V,theta);
		vec lamOutInv(D,fill::zeros);
		for(int i = 0; i < D; i++)
		{
			if(lamOut.at(i) < 1e-10)
				break;
			lamOutInv.at(i) = 1.0/lamOut.at(i);
		}
		V = V.t();
		for(int i = 0; i < d; i++)
		{
			gamLeft.rows(i*D,i*D+D-1) = diagmat(lamOutInv)*U.submat(i*D,0,i*D+D-1,D-1);
			gamRight.rows(i*D,i*D+D-1) = V.submat(0,i*D,D-1,i*D+D-1)*diagmat(lamOutInv);
		}
		lamIn = normalise(s.head(D));
	}

}

template<typename T>
imps<T>::imps(int d, int chi)
	: d_(d), chi_(chi)
{
	for(int i = 0; i < 2; i++)
	{
		gamma_[i].resize(d*chi,chi);
		lambda_[i].resize(chi);

		gamma_[i].zeros();
		lambda_[i].zeros();
	}
	gamma_[0].at(0,0) = 1;
	gamma_[1].at(0,0) = 1;
	lambda_[0].at(0) = 1;
	lambda_[1].at(0) = 1;
}


template<typename T>
void imps<T>::applyTwoSite(const arma::Mat<T>& u, int n)
{
	Mat<T> t(d_*chi_,d_*chi_,fill::zeros);

	for(int i = 0; i < d_; i++)
	{
		for(int j = 0; j < d_; j++)
		{
			for(int k = 0; k < d_; k++)
			{
				for(int l = 0; l < d_; l++)
				{
					Mat<T> tkl = arma::diagmat(lambda_[(n+1)%2])*
						gamma_[n].rows(k*chi_,k*chi_+chi_-1)*
						arma::diagmat(lambda_[n])*gamma_[(n+1)%2].rows(l*chi_,l*chi_+chi_-1)*arma::diagmat(lambda_[(n+1)%2]);
					t.submat(i*chi_,j*chi_,i*chi_+chi_-1,j*chi_+chi_-1) += 
						u.at(idx(i,j),idx(k,l))*tkl;
				}
			}
		}
	}
	reduceDim(d_,chi_,t,gamma_[n],gamma_[(n+1)%2],lambda_[n],lambda_[(n+1)%2]);
}

template<typename T>
T imps<T>::expectationTwoSite(const arma::Mat<T>& h, int n)
{
	T s{0.0};
	for(int i = 0; i < d_; i++)
	{
		for(int j = 0; j < d_; j++)
		{
			Mat<T> Cij = diagmat(lambda_[(n+1)%2])*gamma_[n].rows(i*chi_,i*chi_+chi_-1)*diagmat(lambda_[n])*gamma_[(n+1)%2].rows(j*chi_,j*chi_+chi_-1)*diagmat(lambda_[(n+1)%2]);
			Mat<T> Sij(chi_,chi_,fill::zeros);
			for(int k = 0; k < d_; k++)
			{
				for(int l = 0; l < d_; l++)
				{
					Sij += h.at(idx(i,j),idx(k,l))*diagmat(lambda_[(n+1)%2])*gamma_[n].rows(k*chi_,k*chi_+chi_-1)*diagmat(lambda_[n])*gamma_[(n+1)%2].rows(l*chi_,l*chi_+chi_-1)*diagmat(lambda_[(n+1)%2]);
				}
			}
			s += trace(Cij.t()*Sij);
		}
	}
	return s;
}

template<typename T>
arma::Col<T> imps<T>::applyTMLeft(arma::Col<T>& v)
{
	Col<T> res(chi_*chi_, fill::zeros);
	for(int i = 0; i < chi_*chi_; i++)
	{
		for(int k = 0; k < d_; k++)
		{
			for(int j = 0; j < chi_*chi_; j++)
			{
				res.at(i) += gamma_[0].at(k*chi_+i/chi_,j/chi_)*lambda_[0].at(j/chi_)*bar(gamma_[0].at(k*chi_+i%chi_,j%chi_))*lambda_[0].at(j%chi_)*v.at(j);
			}
		}
	}
	return res;
}
template<typename T>
arma::Row<T> imps<T>::applyTMRight(arma::Row<T>& v)
{
	Row<T> res(chi_*chi_, fill::zeros);
	for(int i = 0; i < chi_*chi_; i++)
	{
		for(int j = 0; j < chi_*chi_; j++)
		{
			for(int k = 0; k < d_; k++)
			{
				res.at(j) += gamma_[0].at(k*chi_+i/chi_,j/chi_)*lambda_[0].at(j/chi_)*bar(lambda_[0].at(k*chi_+i%chi_,j%chi_))*lambda_[0].at(j%chi_)*v.at(i);
			}
		}
	}
	return res;
}
