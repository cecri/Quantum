#include "RandomUnitary.h"
template<class T>
inline T sqr(T a)
{
	return a;
}
arma::cx_mat randomUnitary(int n)
{
	using namespace arma;
	cx_mat Q, R;
	while(qr(Q,R,randn<cx_mat>(n,n)) == false)
	{
	}
	cx_vec r = R.diag();
	cx_mat L = diagmat(r/abs(r));
	return Q*L;
}
