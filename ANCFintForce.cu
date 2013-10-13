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

__global__ void addInternalForceComponent(double* fint, double* strainD_shared, double* strainVec, double* stiffness, Material* materials, double wtl, double betah2, int numElements, int check)
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

//		__shared__ double strainD_shared[12];
//		for(int j=0;j<12;j++) strainD_shared[j] = strainD[j];

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

		factor = factor * betah2;
		for (int j = 0; j < 12; j++) {
			for (int k = 0; k < 12; k++) {
				stiffness[12 * j + k] += strainD_shared[j] * strainD_shared[k]
						* factor;
			}
		}

	}
}

int ANCFSystem::updateInternalForces()
{
	thrust::fill(fint_d.begin(),fint_d.end(),0.0); //Clear internal forces
	thrust::fill(stiffness_d.begin(),stiffness_d.end(),0.0); //Clear internal forces
//	print(stiffness);
//	cin.get();

	for(int j=0;j<pt5.size();j++)
	{
		strainDerivativeUpdate<<<dimGridElement,dimBlockElement>>>(pt5[j],CASTD1(pnew_d),CASTD1(strain_d),CASTD1(strainDerivative_d),CASTD1(Sx_d),CASTM1(materials_d),elements.size());
		addInternalForceComponent<<<dimGridElement,dimBlockElement>>>(CASTD1(fint_d),CASTD1(strainDerivative_d),CASTD1(strain_d),CASTD1(stiffness_d),CASTM1(materials_d),wt5[j],betaHHT*h*h,elements.size(),0);
//		print(stiffness);
//		cin.get();
	}

	for(int j=0;j<pt3.size();j++)
	{
		curvatDerivUpdate<<<dimGridElement,dimBlockElement>>>(pt3[j],CASTD1(pnew_d),CASTD1(strain_d),CASTD1(strainDerivative_d),CASTD1(Sx_d),CASTD1(Sxx_d),CASTM1(materials_d),elements.size());
		addInternalForceComponent<<<dimGridElement,dimBlockElement>>>(CASTD1(fint_d),CASTD1(strainDerivative_d),CASTD1(strain_d),CASTD1(stiffness_d),CASTM1(materials_d),wt3[j],betaHHT*h*h,elements.size(),1);
//		print(stiffness);
//		cin.get();
	}
	//print(stiffness);
	//cin.get();

	return 0;
}
