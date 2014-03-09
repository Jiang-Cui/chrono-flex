#include "include.cuh"
#include "ANCFSystem.cuh"

__device__ int ancf_shape_derivative_x(double* Sx, double x, double a)
{
	double xi = x/a;

	Sx[0] = (6*xi*xi-6*xi)/a;
	Sx[1] = 1-4*xi+3*xi*xi;
	Sx[2] = -(6*xi*xi-6*xi)/a;
	Sx[3] = -2*xi+3*xi*xi;

	return 0;
}

__device__ int ancf_shape_derivative2_x(double* Sxx, double x, double a)
{
	double xi = x/a;

	Sxx[0] = (12*xi-6)/(a*a);
	Sxx[1] = (-4+6*xi)/a;
	Sxx[2] = (6-12*xi)/(a*a);
	Sxx[3] = (-2+6*xi)/a;

	return 0;
}

__global__ void strainDerivativeUpdate(double ptj, double* p, double* strain, double* strainD, double* Sx, Material* materials, int numElements)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		double a = materials[i].l;

		double x = .5*a*(1.+ptj);

		p = &p[12*i];
		strainD = &strainD[12*i];
		Sx = &Sx[4*i];

		ancf_shape_derivative_x(Sx,x,a);

		strain[i] = .5*(((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[0] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[1] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[2]) * Sx[0] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[3] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[4] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[5]) * Sx[1] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[6] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[7] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[8]) * Sx[2] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[9] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[10] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[11]) * Sx[3]-1);

		strainD[0]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[0];
		strainD[1]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[0];
		strainD[2]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[0];
		strainD[3]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[1];
		strainD[4]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[1];
		strainD[5]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[1];
		strainD[6]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[2];
		strainD[7]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[2];
		strainD[8]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[2];
		strainD[9]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[3];
		strainD[10] = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[3];
		strainD[11] = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[3];
	}
}

__global__ void curvatDerivUpdate(double ptj, double* p, double* k, double* ke, double* Sx, double* Sxx, Material* materials, int numElements)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		double a = materials[i].l;
		double x = .5*a*(1.+ptj);

		p = &p[12*i];
		ke = &ke[12*i];
		Sx = &Sx[4*i];
		Sxx = &Sxx[4*i];

		ancf_shape_derivative_x(Sx,x,a);
		ancf_shape_derivative2_x(Sxx,x,a);

		double3 f1;
		double3 rx;
		double3 rxx;

		rx.x = p[0] * Sx[0] + p[3] * Sx[1] + p[6] * Sx[2] + p[9] * Sx[3];
		rx.y = p[1] * Sx[0] + p[4] * Sx[1] + p[7] * Sx[2] + p[10] * Sx[3];
		rx.z = p[2] * Sx[0] + p[5] * Sx[1] + p[8] * Sx[2] + p[11] * Sx[3];

		rxx.x = p[0] * Sxx[0] + p[3] * Sxx[1] + p[6] * Sxx[2] + p[9] * Sxx[3];
		rxx.y = p[1] * Sxx[0] + p[4] * Sxx[1] + p[7] * Sxx[2] + p[10] * Sxx[3];
		rxx.z = p[2] * Sxx[0] + p[5] * Sxx[1] + p[8] * Sxx[2] + p[11] * Sxx[3];

		double g1 = sqrt(rx.x*rx.x+rx.y*rx.y+rx.z*rx.z);
		double g = pow(g1,3);

		f1.x = rx.y * rxx.z - rx.z * rxx.y;
		f1.y = rx.z * rxx.x - rx.x * rxx.z;
		f1.z = rx.x * rxx.y - rx.y * rxx.x;

		double f = sqrt(f1.x*f1.x+f1.y*f1.y+f1.z*f1.z);

		k[i] = f/g;

		double fspecial = -1.0;
		if(f) fspecial = fspecial/f;

		ke[0] = pow(g, -0.2e1) * (g * (fspecial * f1.y * (-Sx[0] * rxx.z - rx.z * Sxx[0]) + fspecial * f1.z * (Sx[0] * rxx.y + rx.y * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[0]);
		ke[1] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (Sx[0] * rxx.z + rx.z * Sxx[0]) + fspecial * f1.z * (-Sx[0] * rxx.x - rx.x * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[0]);
		ke[2] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (-Sx[0] * rxx.y - rx.y * Sxx[0]) + fspecial * f1.y * (Sx[0] * rxx.x + rx.x * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[0]);
		ke[3] = pow(g, -0.2e1) * (g * (fspecial * f1.y * (-Sx[1] * rxx.z - rx.z * Sxx[1]) + fspecial * f1.z * (Sx[1] * rxx.y + rx.y * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[1]);
		ke[4] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (Sx[1] * rxx.z + rx.z * Sxx[1]) + fspecial * f1.z * (-Sx[1] * rxx.x - rx.x * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[1]);
		ke[5] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (-Sx[1] * rxx.y - rx.y * Sxx[1]) + fspecial * f1.y * (Sx[1] * rxx.x + rx.x * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[1]);
		ke[6] = pow(g, -0.2e1) * (g * (fspecial * f1.y * (-Sx[2] * rxx.z - rx.z * Sxx[2]) + fspecial * f1.z * (Sx[2] * rxx.y + rx.y * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[2]);
		ke[7] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (Sx[2] * rxx.z + rx.z * Sxx[2]) + fspecial * f1.z * (-Sx[2] * rxx.x - rx.x * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[2]);
		ke[8] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (-Sx[2] * rxx.y - rx.y * Sxx[2]) + fspecial * f1.y * (Sx[2] * rxx.x + rx.x * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[2]);
		ke[9] = pow(g, -0.2e1) * (g * (fspecial * f1.y * (-Sx[3] * rxx.z - rx.z * Sxx[3]) + fspecial * f1.z * (Sx[3] * rxx.y + rx.y * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[3]);
		ke[10] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (Sx[3] * rxx.z + rx.z * Sxx[3]) + fspecial * f1.z * (-Sx[3] * rxx.x - rx.x * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[3]);
		ke[11] = pow(g, -0.2e1) * (g * (fspecial * f1.x * (-Sx[3] * rxx.y - rx.y * Sxx[3]) + fspecial * f1.y * (Sx[3] * rxx.x + rx.x * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[3]);
	}
}
/* Radu's code
__global__ void curvatDerivUpdate(double ptj, double* p, double* k, double* ke, double* Sx, double* Sxx, Material* materials, int numElements)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		double L = materials[i].l;
		double x = .5*L*(1.+ptj);

		p = &p[12*i];
		ke = &ke[12*i];
		Sx = &Sx[4*i];
		Sxx = &Sxx[4*i];

		double e1  = p[0];
		double e2  = p[1];
		double e3  = p[2];
		double e4  = p[3];
		double e5  = p[4];
		double e6  = p[5];

		double e7  = p[6];
		double e8  = p[7];
		double e9  = p[8];
		double e10 = p[9];
		double e11 = p[10];
		double e12 = p[11];

//		double Sx[4];
//		double Sxx[4];

		ancf_shape_derivative_x(Sx, x, L);
		ancf_shape_derivative2_x(Sxx, x, L);

		double rx[3];
		double rxx[3];

		rx[0] = Sx[0]*e1 + Sx[1]*e4 + Sx[2]*e7 + Sx[3]*e10;
		rx[1] = Sx[0]*e2 + Sx[1]*e5 + Sx[2]*e8 + Sx[3]*e11;
		rx[2] = Sx[0]*e3 + Sx[1]*e6 + Sx[2]*e9 + Sx[3]*e12;

		rxx[0] = Sxx[0]*e1 + Sxx[1]*e4 + Sxx[2]*e7 + Sxx[3]*e10;
		rxx[1] = Sxx[0]*e2 + Sxx[1]*e5 + Sxx[2]*e8 + Sxx[3]*e11;
		rxx[2] = Sxx[0]*e3 + Sxx[1]*e6 + Sxx[2]*e9 + Sxx[3]*e12;

		double rx_rx   =  rx[0] *  rx[0] +  rx[1] *  rx[1] +  rx[2] *  rx[2];
		double rx_rxx  =  rx[0] * rxx[0] +  rx[1] * rxx[1] +  rx[2] * rxx[2];
		double rxx_rxx = rxx[0] * rxx[0] + rxx[1] * rxx[1] + rxx[2] * rxx[2];

		// Define v = rx X rxx
		double v[3];

		v[0] = rx[1]*rxx[2]-rx[2]*rxx[1];
		v[1] = -rx[0]*rxx[2]+rx[2]*rxx[0];
		v[2] = rx[0]*rxx[1]-rx[1]*rxx[0];

		// Calculate the squared norm of v using the Lagrange identity
		double v_v = rx_rx * rxx_rxx - rx_rxx * rx_rxx;

		// Define some temporary variables
		double inv_rx_rx_cubed = 1.0 / (rx_rx * rx_rx * rx_rx);
		double coef = 3 * v_v / rx_rx;

		// Calculate k
		k[i] = sqrt(v_v * inv_rx_rx_cubed);
		printf("K = %f\n", k[i]);

		// Calculate the integrand for Qk (this is k * k_e)
		inv_rx_rx_cubed = -inv_rx_rx_cubed/k[i]; // divide by k to get k_e
		ke[0]  = -inv_rx_rx_cubed * ( -(Sx[0]*rxx[1]-Sxx[0]*rx[1])*v[2]+(Sx[0]*rxx[2]-Sxx[0]*rx[2])*v[1] + coef*Sx[0]*rx[0] );
		ke[1]  = -inv_rx_rx_cubed * (  (Sx[0]*rxx[0]-Sxx[0]*rx[0])*v[2]-(Sx[0]*rxx[2]-Sxx[0]*rx[2])*v[0] + coef*Sx[0]*rx[1] );
		ke[2]  = -inv_rx_rx_cubed * ( -(Sx[0]*rxx[0]-Sxx[0]*rx[0])*v[1]+(Sx[0]*rxx[1]-Sxx[0]*rx[1])*v[0] + coef*Sx[0]*rx[2] );
		ke[3]  = -inv_rx_rx_cubed * ( -(Sx[1]*rxx[1]-Sxx[1]*rx[1])*v[2]+(Sx[1]*rxx[2]-Sxx[1]*rx[2])*v[1] + coef*Sx[1]*rx[0] );
		ke[4]  = -inv_rx_rx_cubed * (  (Sx[1]*rxx[0]-Sxx[1]*rx[0])*v[2]-(Sx[1]*rxx[2]-Sxx[1]*rx[2])*v[0] + coef*Sx[1]*rx[1] );
		ke[5]  = -inv_rx_rx_cubed * ( -(Sx[1]*rxx[0]-Sxx[1]*rx[0])*v[1]+(Sx[1]*rxx[1]-Sxx[1]*rx[1])*v[0] + coef*Sx[1]*rx[2] );
		ke[6]  = -inv_rx_rx_cubed * ( -(Sx[2]*rxx[1]-Sxx[2]*rx[1])*v[2]+(Sx[2]*rxx[2]-Sxx[2]*rx[2])*v[1] + coef*Sx[2]*rx[0] );
		ke[7]  = -inv_rx_rx_cubed * (  (Sx[2]*rxx[0]-Sxx[2]*rx[0])*v[2]-(Sx[2]*rxx[2]-Sxx[2]*rx[2])*v[0] + coef*Sx[2]*rx[1] );
		ke[8]  = -inv_rx_rx_cubed * ( -(Sx[2]*rxx[0]-Sxx[2]*rx[0])*v[1]+(Sx[2]*rxx[1]-Sxx[2]*rx[1])*v[0] + coef*Sx[2]*rx[2] );
		ke[9]  = -inv_rx_rx_cubed * ( -(Sx[3]*rxx[1]-Sxx[3]*rx[1])*v[2]+(Sx[3]*rxx[2]-Sxx[3]*rx[2])*v[1] + coef*Sx[3]*rx[0] );
		ke[10] = -inv_rx_rx_cubed * (  (Sx[3]*rxx[0]-Sxx[3]*rx[0])*v[2]-(Sx[3]*rxx[2]-Sxx[3]*rx[2])*v[0] + coef*Sx[3]*rx[1] );
		ke[11] = -inv_rx_rx_cubed * ( -(Sx[3]*rxx[0]-Sxx[3]*rx[0])*v[1]+(Sx[3]*rxx[1]-Sxx[3]*rx[1])*v[0] + coef*Sx[3]*rx[2] );

		for(int idx = 0;idx<12;idx++) printf("Ke[%d] = %f\n", idx, ke[idx]);
		printf("\n");
	}
}
*/
__global__ void addInternalForceComponent(double* fint, double* strainD_shared, double* strainVec, double* stiffness, Material* materials, double wtl, double betah2, int numElements, int check,int updateLhs)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		double strain = strainVec[i];
		Material material = materials[i];
		double E = material.E;
		double a = material.l;
		double r = material.r;
		double A = PI*r*r;
		double I = .25*PI*r*r*r*r;

		fint = &fint[12*i];
		strainD_shared = &strainD_shared[12*i];
		stiffness = &stiffness[12*12*i];
		double factor = wtl*A*E*a*.5;
		if(check) factor = wtl*I*E*a*.5;

		fint[0] += factor * strain * strainD_shared[0];
		fint[1] += factor * strain * strainD_shared[1];
		fint[2] += factor * strain * strainD_shared[2];
		fint[3] += factor * strain * strainD_shared[3];
		fint[4] += factor * strain * strainD_shared[4];
		fint[5] += factor * strain * strainD_shared[5];
		fint[6] += factor * strain * strainD_shared[6];
		fint[7] += factor * strain * strainD_shared[7];
		fint[8] += factor * strain * strainD_shared[8];
		fint[9] += factor * strain * strainD_shared[9];
		fint[10] += factor * strain * strainD_shared[10];
		fint[11] += factor * strain * strainD_shared[11];

		if(updateLhs)
		{
			factor = factor * betah2;
			for (int j = 0; j < 12; j++) {
				for (int k = 0; k < 12; k++) {
					stiffness[12 * j + k] += strainD_shared[j] * strainD_shared[k] * factor;
				}
			}
		}

	}
}

__global__ void addMass(double* stiffness, Material* materials, int numElements)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		Material material = materials[i];
		double l = material.l;
		double r = material.r;
		double A = PI*r*r;
		double rho = material.rho;
		int j = 0;
		int k = 0;

		stiffness = &stiffness[12*12*i];

		j = 0;
		k = 0;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 0;
		k = 3;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 0;
		k = 6;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 0;
		k = 9;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 1;
		k = 1;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 1;
		k = 4;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 1;
		k = 7;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 1;
		k = 10;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 2;
		k = 2;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 2;
		k = 5;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 2;
		k = 8;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 2;
		k = 11;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 3;
		k = 0;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 3;
		k = 3;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;

		j = 3;
		k = 6;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 3;
		k = 9;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 4;
		k = 1;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 4;
		k = 4;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;

		j = 4;
		k = 7;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 4;
		k = 10;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 5;
		k = 2;
		stiffness[12 * k + j] += 0.11e2 / 0.210e3 * rho * A * l * l;

		j = 5;
		k = 5;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;

		j = 5;
		k = 8;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 5;
		k = 11;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 6;
		k = 0;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 6;
		k = 3;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 6;
		k = 6;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 6;
		k = 9;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 7;
		k = 1;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 7;
		k = 4;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 7;
		k = 7;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 7;
		k = 10;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 8;
		k = 2;
		stiffness[12 * k + j] += 0.9e1 / 0.70e2 * rho * A * l;

		j = 8;
		k = 5;
		stiffness[12 * k + j] += 0.13e2 / 0.420e3 * rho * A * l * l;

		j = 8;
		k = 8;
		stiffness[12 * k + j] += 0.13e2 / 0.35e2 * rho * A * l;

		j = 8;
		k = 11;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 9;
		k = 0;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 9;
		k = 3;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 9;
		k = 6;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 9;
		k = 9;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;

		j = 10;
		k = 1;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 10;
		k = 4;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 10;
		k = 7;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 10;
		k = 10;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;

		j = 11;
		k = 2;
		stiffness[12 * k + j] += -0.13e2 / 0.420e3 * rho * A * l * l;

		j = 11;
		k = 5;
		stiffness[12 * k + j] += -rho * A * l * l * l / 0.140e3;

		j = 11;
		k = 8;
		stiffness[12 * k + j] += -0.11e2 / 0.210e3 * rho * A * l * l;

		j = 11;
		k = 11;
		stiffness[12 * k + j] += rho * A * l * l * l / 0.105e3;
	}
}

int ANCFSystem::resetLeftHandSideMatrix()
{
	// populate the lhs with only the mass, must be done before updating internal forces!
	thrust::fill_n(lhs_d.begin(),elements.size()*12*12,0.0); //Clear the matrix
	addMass<<<dimGridElement,dimBlockElement>>>(CASTD1(lhs_d),CASTM1(materials_d),elements.size()); // add terms from mass
	return 0;
}

int ANCFSystem::updateInternalForces()
{
	int updateLhs = 1;
	thrust::fill(fint_d.begin(),fint_d.end(),0.0); //Clear internal forces

	ANCFSystem::resetLeftHandSideMatrix(); // populate the lhs with only the mass

	for(int j=0;j<pt5.size();j++)
	{
		strainDerivativeUpdate<<<dimGridElement,dimBlockElement>>>(pt5[j],CASTD1(pnew_d),CASTD1(strain_d),CASTD1(strainDerivative_d),CASTD1(Sx_d),CASTM1(materials_d),elements.size());
		addInternalForceComponent<<<dimGridElement,dimBlockElement>>>(CASTD1(fint_d),CASTD1(strainDerivative_d),CASTD1(strain_d),CASTD1(lhs_d),CASTM1(materials_d),wt5[j],betaHHT*h*h,elements.size(),0,updateLhs);
	}

	for(int j=0;j<pt3.size();j++)
	{
		curvatDerivUpdate<<<dimGridElement,dimBlockElement>>>(pt3[j],CASTD1(pnew_d),CASTD1(strain_d),CASTD1(strainDerivative_d),CASTD1(Sx_d),CASTD1(Sxx_d),CASTM1(materials_d),elements.size());
		addInternalForceComponent<<<dimGridElement,dimBlockElement>>>(CASTD1(fint_d),CASTD1(strainDerivative_d),CASTD1(strain_d),CASTD1(lhs_d),CASTM1(materials_d),wt3[j],betaHHT*h*h,elements.size(),1,updateLhs);
	}

	return 0;
}
