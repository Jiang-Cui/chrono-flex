#include "include.cuh"

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

//		factor = factor*betah2;
//		for(int j=0;j<12;j++)
//		{
//			for(int k=0;k<12;k++)
//			{
//				stiffness[12*j+k] += strainD_shared[j]*strainD_shared[k]*factor;
//			}
//		}


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

//int ancf_shape_derivative_x_CPU(double* Sx, double x, double a)
//{
//	double xi = x/a;
//
//	Sx[0] = (6*xi*xi-6*xi)/a;
//	Sx[1] = 1-4*xi+3*xi*xi;
//	Sx[2] = -(6*xi*xi-6*xi)/a;
//	Sx[3] = -2*xi+3*xi*xi;
//
//	return 0;
//}
//
//int ancf_shape_derivative2_x_CPU(double* Sxx, double x, double a)
//{
//	double xi = x/a;
//
//	Sxx[0] = (12*xi-6)/(a*a);
//	Sxx[1] = (-4+6*xi)/a;
//	Sxx[2] = (6-12*xi)/(a*a);
//	Sxx[3] = (-2+6*xi)/a;
//
//	return 0;
//}
//
//void strainDerivativeUpdate_CPU(double ptj, double* pAll, double* strain, double* strainDAll, double* SxAll, Material* materials, int numElements)
//{
//	for(int i=0;i<numElements;i++)
//	{
//		double a = materials[i].l;
//
//		double x = .5*a*(1.+ptj);
//
//		double* p = &pAll[12*i];
//		double* strainD = &strainDAll[12*i];
//		double* Sx = &SxAll[4*i];
//
//		ancf_shape_derivative_x_CPU(Sx,x,a);
//
//		strain[i] = .5*(((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[0] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[1] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[2]) * Sx[0] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[3] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[4] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[5]) * Sx[1] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[6] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[7] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[8]) * Sx[2] + ((Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * p[9] + (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * p[10] + (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * p[11]) * Sx[3]-1);
//
//		strainD[0]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[0];
//		strainD[1]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[0];
//		strainD[2]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[0];
//		strainD[3]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[1];
//		strainD[4]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[1];
//		strainD[5]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[1];
//		strainD[6]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[2];
//		strainD[7]  = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[2];
//		strainD[8]  = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[2];
//		strainD[9]  = (Sx[0] * p[0] + Sx[1] * p[3] + Sx[2] * p[6] + Sx[3] * p[9]) * Sx[3];
//		strainD[10] = (Sx[0] * p[1] + Sx[1] * p[4] + Sx[2] * p[7] + Sx[3] * p[10]) * Sx[3];
//		strainD[11] = (Sx[0] * p[2] + Sx[1] * p[5] + Sx[2] * p[8] + Sx[3] * p[11]) * Sx[3];
//	}
//}
//
//void curvatDerivUpdate_CPU(double ptj, double* pAll, double* k, double* keAll, double* SxAll, double* SxxAll, Material* materials, int numElements)
//{
//	for(int i=0;i<numElements;i++)
//	{
//		double a = materials[i].l;
//		double x = .5*a*(1.+ptj);
//
//		double* p = &pAll[12*i];
//		double* ke = &keAll[12*i];
//		double* Sx = &SxAll[4*i];
//		double* Sxx = &SxxAll[4*i];
//
//		ancf_shape_derivative_x_CPU(Sx,x,a);
//		ancf_shape_derivative2_x_CPU(Sxx,x,a);
//
//		double3 f1;
//		double3 rx;
//		double3 rxx;
//
//		rx.x = p[0] * Sx[0] + p[3] * Sx[1] + p[6] * Sx[2] + p[9] * Sx[3];
//		rx.y = p[1] * Sx[0] + p[4] * Sx[1] + p[7] * Sx[2] + p[10] * Sx[3];
//		rx.z = p[2] * Sx[0] + p[5] * Sx[1] + p[8] * Sx[2] + p[11] * Sx[3];
//
//		rxx.x = p[0] * Sxx[0] + p[3] * Sxx[1] + p[6] * Sxx[2] + p[9] * Sxx[3];
//		rxx.y = p[1] * Sxx[0] + p[4] * Sxx[1] + p[7] * Sxx[2] + p[10] * Sxx[3];
//		rxx.z = p[2] * Sxx[0] + p[5] * Sxx[1] + p[8] * Sxx[2] + p[11] * Sxx[3];
//
//		double g1 = sqrt(rx.x*rx.x+rx.y*rx.y+rx.z*rx.z);
//		double g = g1*g1*g1;
//
//		f1.x = rx.y * rxx.z - rx.z * rxx.y;
//		f1.y = rx.z * rxx.x - rx.x * rxx.z;
//		f1.z = rx.x * rxx.y - rx.y * rxx.x;
//
//		double f = sqrt(f1.x*f1.x+f1.y*f1.y+f1.z*f1.z);
//
//		k[i] = f/g;
//
//		double fspecial = -1.0;
//		if(f) fspecial = fspecial/f;
//
//		double g2inv = 1/(g*g);
//
//		ke[0] = g2inv * (g * (fspecial * f1.y * (-Sx[0] * rxx.z - rx.z * Sxx[0]) + fspecial * f1.z * (Sx[0] * rxx.y + rx.y * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[0]);
//		ke[1] = g2inv * (g * (fspecial * f1.x * (Sx[0] * rxx.z + rx.z * Sxx[0]) + fspecial * f1.z * (-Sx[0] * rxx.x - rx.x * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[0]);
//		ke[2] = g2inv * (g * (fspecial * f1.x * (-Sx[0] * rxx.y - rx.y * Sxx[0]) + fspecial * f1.y * (Sx[0] * rxx.x + rx.x * Sxx[0])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[0]);
//		ke[3] = g2inv * (g * (fspecial * f1.y * (-Sx[1] * rxx.z - rx.z * Sxx[1]) + fspecial * f1.z * (Sx[1] * rxx.y + rx.y * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[1]);
//		ke[4] = g2inv * (g * (fspecial * f1.x * (Sx[1] * rxx.z + rx.z * Sxx[1]) + fspecial * f1.z * (-Sx[1] * rxx.x - rx.x * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[1]);
//		ke[5] = g2inv * (g * (fspecial * f1.x * (-Sx[1] * rxx.y - rx.y * Sxx[1]) + fspecial * f1.y * (Sx[1] * rxx.x + rx.x * Sxx[1])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[1]);
//		ke[6] = g2inv * (g * (fspecial * f1.y * (-Sx[2] * rxx.z - rx.z * Sxx[2]) + fspecial * f1.z * (Sx[2] * rxx.y + rx.y * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[2]);
//		ke[7] = g2inv * (g * (fspecial * f1.x * (Sx[2] * rxx.z + rx.z * Sxx[2]) + fspecial * f1.z * (-Sx[2] * rxx.x - rx.x * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[2]);
//		ke[8] = g2inv * (g * (fspecial * f1.x * (-Sx[2] * rxx.y - rx.y * Sxx[2]) + fspecial * f1.y * (Sx[2] * rxx.x + rx.x * Sxx[2])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[2]);
//		ke[9] = g2inv * (g * (fspecial * f1.y * (-Sx[3] * rxx.z - rx.z * Sxx[3]) + fspecial * f1.z * (Sx[3] * rxx.y + rx.y * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[0] + 0.3e1 * g1 * Sx[1] * p[3] + 0.3e1 * g1 * Sx[2] * p[6] + 0.3e1 * g1 * Sx[3] * p[9]) * Sx[3]);
//		ke[10] = g2inv * (g * (fspecial * f1.x * (Sx[3] * rxx.z + rx.z * Sxx[3]) + fspecial * f1.z * (-Sx[3] * rxx.x - rx.x * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[1] + 0.3e1 * g1 * Sx[1] * p[4] + 0.3e1 * g1 * Sx[2] * p[7] + 0.3e1 * g1 * Sx[3] * p[10]) * Sx[3]);
//		ke[11] = g2inv * (g * (fspecial * f1.x * (-Sx[3] * rxx.y - rx.y * Sxx[3]) + fspecial * f1.y * (Sx[3] * rxx.x + rx.x * Sxx[3])) - f * (0.3e1 * g1 * Sx[0] * p[2] + 0.3e1 * g1 * Sx[1] * p[5] + 0.3e1 * g1 * Sx[2] * p[8] + 0.3e1 * g1 * Sx[3] * p[11]) * Sx[3]);
//	}
//}
//
//void addInternalForceComponent_CPU(double* fintAll, double* strainDAll, double* strainVec, double* stiffnessAll, Material* materials, double wtl, double betah2, int numElements, int check)
//{
//	for(int i=0;i<numElements;i++)
//	{
//		double strain = strainVec[i];
//		Material material = materials[i];
//		double E = material.E;
//		double a = material.l;
//		double r = material.r;
//		double A = PI*r*r;
//		double I = .25*PI*r*r*r*r;
//
//		double* fint = &fintAll[12*i];
//		double* strainD = &strainDAll[12*i];
//		double* stiffness = &stiffnessAll[12*12*i];
//		double factor = wtl*A*E*a*.5;
//		if(check) factor = wtl*I*E*a*.5;
//
////		__shared__ double strainD_shared[12];
////		for(int j=0;j<12;j++) strainD_shared[j] = strainD[j];
//
//		fint[0] += factor * strain * strainD[0];
//		fint[1] += factor * strain * strainD[1];
//		fint[2] += factor * strain * strainD[2];
//		fint[3] += factor * strain * strainD[3];
//		fint[4] += factor * strain * strainD[4];
//		fint[5] += factor * strain * strainD[5];
//		fint[6] += factor * strain * strainD[6];
//		fint[7] += factor * strain * strainD[7];
//		fint[8] += factor * strain * strainD[8];
//		fint[9] += factor * strain * strainD[9];
//		fint[10] += factor * strain * strainD[10];
//		fint[11] += factor * strain * strainD[11];
//
//		factor = factor*betah2;
//		for(int j=0;j<12;j++)
//		{
//			for(int k=0;k<12;k++)
//			{
//				stiffness[12*j+k] += strainD[j]*strainD[k]*factor;
//			}
//		}
//	}
//}
//
//int ANCFSystem::updateInternalForcesCPU()
//{
//	pnew_h = pnew_d;
//	thrust::fill(fint_h.begin(),fint_h.end(),0.0); //Clear internal forces
//	thrust::fill(stiffness_h.begin(),stiffness_h.end(),0.0); //Clear internal forces
////	print(stiffness);
////	cin.get();
//
//	for(int j=0;j<pt5.size();j++)
//	{
//		strainDerivativeUpdate_CPU(pt5[j],CASTD1(pnew_h),CASTD1(strain_h),CASTD1(strainDerivative_h),CASTD1(Sx_h),CASTM1(materials),elements.size());
//		addInternalForceComponent_CPU(CASTD1(fint_h),CASTD1(strainDerivative_h),CASTD1(strain_h),CASTD1(stiffness_h),CASTM1(materials),wt5[j],betaHHT*h*h,elements.size(),0);
////		print(stiffness);
////		cin.get();
//	}
//
//	for(int j=0;j<pt3.size();j++)
//	{
//		curvatDerivUpdate_CPU(pt3[j],CASTD1(pnew_h),CASTD1(strain_h),CASTD1(strainDerivative_h),CASTD1(Sx_h),CASTD1(Sxx_h),CASTM1(materials),elements.size());
//		addInternalForceComponent_CPU(CASTD1(fint_h),CASTD1(strainDerivative_h),CASTD1(strain_h),CASTD1(stiffness_h),CASTM1(materials),wt3[j],betaHHT*h*h,elements.size(),1);
////		print(stiffness);
////		cin.get();
//	}
//	fint_d = fint_h;
//	stiffness_d = stiffness_h;
//
//	return 0;
//}
//
//void strainDerivativeUpdate_ARMA(double ptj, double* pAll, double* strain, double* strainDAll, double* Sx, Material* materials, int numElements)
//{
//	for(int i=0;i<numElements;i++)
//	{
//		double l = materials[i].l;
//		double x = .5*l*(1.+ptj);
//
//		double* p = &pAll[12*i];
//		double* strainD = &strainDAll[12*i];
//
//		double xx = x/l;
//
//		double s1 = (6.*xx*xx-6.*xx)/l;
//		double s2 = 1.-4.*xx+3.*xx*xx;
//		double s3 = (6. * xx - 6. * xx * xx) / l;
//		double s4 = -2. * xx + 3. * xx * xx;
//
//		mat S = zeros(1, 4);
//		mat Sd = zeros(3, 12);
//
//		S(0, 0) = s1;
//		S(0, 1) = s2;
//		S(0, 2) = s3;
//		S(0, 3) = s4;
//
//		Sd(0, 0) = s1;
//		Sd(1, 1) = s1;
//		Sd(2, 2) = s1;
//		Sd(0, 3) = s2;
//		Sd(1, 4) = s2;
//		Sd(2, 5) = s2;
//		Sd(0, 6) = s3;
//		Sd(1, 7) = s3;
//		Sd(2, 8) = s3;
//		Sd(0, 9) = s4;
//		Sd(1, 10) = s4;
//		Sd(2, 11) = s4;
//
//		mat d = zeros(4, 3);
//		d(0, 0) = p[0];
//		d(0, 1) = p[1];
//		d(0, 2) = p[2];
//		d(1, 0) = p[3];
//		d(1, 1) = p[4];
//		d(1, 2) = p[5];
//		d(2, 0) = p[6];
//		d(2, 1) = p[7];
//		d(2, 2) = p[8];
//		d(3, 0) = p[9];
//		d(3, 1) = p[10];
//		d(3, 2) = p[11];
//
//		mat strainMat = .5 * (S * d * trans(d) * trans(S) - 1.0);
//
//		double strainScalar = strainMat(0, 0);
//		strain[i] = strainScalar;
//
//		mat strainDeriv = S * d * Sd;
//
//		for (int i = 0; i < 12; i++)
//		{
//			strainD[i] = strainDeriv(0, i);
//		}
//	}
//}
//
//mat crossD(mat A, mat B)
//{
//	mat C = zeros(3,A.n_cols);
//	for(int i=0;i<A.n_cols;i++)
//	{
//		C.col(i) = cross(A.col(i),B.col(i));
//	}
//
//	return C;
//}
//
//void curvatDerivUpdate_ARMA(double ptj, double* pAll, double* k, double* keAll, double* Sx, double* Sxx, Material* materials, int numElements)
//{
//	for(int i=0;i<numElements;i++)
//	{
//		double l = materials[i].l;
//		double x = .5*l*(1.+ptj);
//
//		double* p = &pAll[12*i];
//		double* ke = &keAll[12*i];
//		double xx = x / l;
//
//		double s1 = (6. * xx * xx - 6. * xx) / l;
//		double s2 = 1. - 4. * xx + 3. * xx * xx;
//		double s3 = (6. * xx - 6. * xx * xx) / l;
//		double s4 = -2. * xx + 3. * xx * xx;
//
//		double ss1 = (12. * xx - 6.) / (l * l);
//		double ss2 = (6. * xx - 4.) / l;
//		double ss3 = (6. - 12. * xx) / (l * l);
//		double ss4 = (6. * xx - 2.) / l;
//
//		mat S = zeros(1, 4);
//		mat SS = zeros(1, 4);
//		mat Sd1 = zeros(3, 12);
//		mat Sd2 = zeros(3, 12);
//
//		S(0, 0) = s1;
//		S(0, 1) = s2;
//		S(0, 2) = s3;
//		S(0, 3) = s4;
//		SS(0, 0) = ss1;
//		SS(0, 1) = ss2;
//		SS(0, 2) = ss3;
//		SS(0, 3) = ss4;
//
//		Sd1(0, 0) = s1;
//		Sd1(1, 1) = s1;
//		Sd1(2, 2) = s1;
//		Sd1(0, 3) = s2;
//		Sd1(1, 4) = s2;
//		Sd1(2, 5) = s2;
//		Sd1(0, 6) = s3;
//		Sd1(1, 7) = s3;
//		Sd1(2, 8) = s3;
//		Sd1(0, 9) = s4;
//		Sd1(1, 10) = s4;
//		Sd1(2, 11) = s4;
//
//		Sd2(0, 0) = ss1;
//		Sd2(1, 1) = ss1;
//		Sd2(2, 2) = ss1;
//		Sd2(0, 3) = ss2;
//		Sd2(1, 4) = ss2;
//		Sd2(2, 5) = ss2;
//		Sd2(0, 6) = ss3;
//		Sd2(1, 7) = ss3;
//		Sd2(2, 8) = ss3;
//		Sd2(0, 9) = ss4;
//		Sd2(1, 10) = ss4;
//		Sd2(2, 11) = ss4;
//
//		mat d = zeros(4, 3);
//		d(0, 0) = p[0];
//		d(0, 1) = p[1];
//		d(0, 2) = p[2];
//		d(1, 0) = p[3];
//		d(1, 1) = p[4];
//		d(1, 2) = p[5];
//		d(2, 0) = p[6];
//		d(2, 1) = p[7];
//		d(2, 2) = p[8];
//		d(3, 0) = p[9];
//		d(3, 1) = p[10];
//		d(3, 2) = p[11];
//
//		mat r_x = trans(d) * trans(S);
//		mat r_xx = trans(d) * trans(SS);
//
//		mat f1 = cross(r_x, r_xx);
//		double f = arma::norm(f1, 2);
//		double g1 = norm(r_x, 2);
//		double g = pow(g1, 3);
//
//		double kScalar = f / g;
//		k[i] = kScalar;
//
//		mat g_e = 3.0 * g1 * S * d * Sd1;
//
//		mat r_xxrep = r_xx * ones(1, 12);
//		mat r_xrep = r_x * ones(1, 12);
//
//		mat fe1 = crossD(Sd1, r_xxrep) + crossD(r_xrep, Sd2);
//
//		double fspecial = f;
//		if (!f) fspecial = 1.0;
//		mat f_e = (1 / fspecial) * trans(f1) * fe1;
//		mat k_e = (g * f_e - f * g_e) / (pow(g, 2));
//
//		for (int i = 0; i < 12; i++)
//		{
//			ke[i] = k_e(0, i);
//		}
//	}
//}
//
//int ANCFSystem::updateInternalForcesARMA()
//{
//	pnew_h = pnew_d;
//	thrust::fill(fint_h.begin(),fint_h.end(),0.0); //Clear internal forces
//	thrust::fill(stiffness_h.begin(),stiffness_h.end(),0.0); //Clear internal forces
////	print(stiffness);
////	cin.get();
//
//	for(int j=0;j<pt5.size();j++)
//	{
//		strainDerivativeUpdate_CPU(pt5[j],CASTD1(pnew_h),CASTD1(strain_h),CASTD1(strainDerivative_h),CASTD1(Sx_h),CASTM1(materials),elements.size());
//		addInternalForceComponent_CPU(CASTD1(fint_h),CASTD1(strainDerivative_h),CASTD1(strain_h),CASTD1(stiffness_h),CASTM1(materials),wt5[j],betaHHT*h*h,elements.size(),0);
////		print(stiffness);
////		cin.get();
//	}
//
//	for(int j=0;j<pt3.size();j++)
//	{
//		curvatDerivUpdate_CPU(pt3[j],CASTD1(pnew_h),CASTD1(strain_h),CASTD1(strainDerivative_h),CASTD1(Sx_h),CASTD1(Sxx_h),CASTM1(materials),elements.size());
//		addInternalForceComponent_CPU(CASTD1(fint_h),CASTD1(strainDerivative_h),CASTD1(strain_h),CASTD1(stiffness_h),CASTM1(materials),wt3[j],betaHHT*h*h,elements.size(),1);
////		print(stiffness);
////		cin.get();
//	}
//
//	fint_d = fint_h;
//	stiffness_d = stiffness_h;
//
//	return 0;
//}
///*
//#include "include.cuh"
//
//__device__ double calculateStrainDerivative(double* p, double* strainD, double x, double l)
//{
//	double xx = x/l;
//
//	double s1 = (6*xx*xx-6*xx)/l;
//	double s2 = 1-4*xx+3*xx*xx;
//	double s3 = (6*xx-6*xx*xx)/l;
//	double s4 = -2*xx+3*xx*xx;
//
//	double strain = s3 * p[8] * s4 * p[11] + s1 * s1 * p[0] * p[0] / 0.2e1 + s1 * s1 * p[1] * p[1] / 0.2e1 + s1 * s1 * p[2] * p[2] / 0.2e1 + s2 * s2 * p[3] * p[3] / 0.2e1 + s2 * s2 * p[4] * p[4] / 0.2e1 + s2 * s2 * p[5] * p[5] / 0.2e1 + s3 * s3 * p[6] * p[6] / 0.2e1 + s3 * s3 * p[7] * p[7] / 0.2e1 + s3 * s3 * p[8] * p[8] / 0.2e1 + s4 * s4 * p[9] * p[9] / 0.2e1 + s4 * s4 * p[10] * p[10] / 0.2e1 + s4 * s4 * p[11] * p[11] / 0.2e1 - 0.1e1 / 0.2e1 + s1 * p[0] * s2 * p[3] + s1 * p[0] * s3 * p[6] + s1 * p[0] * s4 * p[9] + s1 * p[1] * s2 * p[4] + s1 * p[1] * s3 * p[7] + s1 * p[1] * s4 * p[10] + s1 * p[2] * s2 * p[5] + s1 * p[2] * s3 * p[8] + s1 * p[2] * s4 * p[11] + s2 * p[3] * s3 * p[6] + s2 * p[3] * s4 * p[9] + s2 * p[4] * s3 * p[7] + s2 * p[4] * s4 * p[10] + s2 * p[5] * s3 * p[8] + s2 * p[5] * s4 * p[11] + s3 * p[6] * s4 * p[9] + s3 * p[7] * s4 * p[10];
//
//	strainD[0] = (s1 * p[0] + s2 * p[3] + s3 * p[6] + s4 * p[9]) * s1;
//	strainD[1] = (s1 * p[1] + s2 * p[4] + s3 * p[7] + s4 * p[10]) * s1;
//	strainD[2] = (s1 * p[2] + s2 * p[5] + s3 * p[8] + s4 * p[11]) * s1;
//	strainD[3] = (s1 * p[0] + s2 * p[3] + s3 * p[6] + s4 * p[9]) * s2;
//	strainD[4] = (s1 * p[1] + s2 * p[4] + s3 * p[7] + s4 * p[10]) * s2;
//	strainD[5] = (s1 * p[2] + s2 * p[5] + s3 * p[8] + s4 * p[11]) * s2;
//	strainD[6] = (s1 * p[0] + s2 * p[3] + s3 * p[6] + s4 * p[9]) * s3;
//	strainD[7] = (s1 * p[1] + s2 * p[4] + s3 * p[7] + s4 * p[10]) * s3;
//	strainD[8] = (s1 * p[2] + s2 * p[5] + s3 * p[8] + s4 * p[11]) * s3;
//	strainD[9] = (s1 * p[0] + s2 * p[3] + s3 * p[6] + s4 * p[9]) * s4;
//	strainD[10] = (s1 * p[1] + s2 * p[4] + s3 * p[7] + s4 * p[10]) * s4;
//	strainD[11] = (s1 * p[2] + s2 * p[5] + s3 * p[8] + s4 * p[11]) * s4;
//
//	return strain;
//}
//
//__device__ double calculateCurvatureDerivative(double* p, double* curvatureDeriv, double x, double l)
//{
//	double xx = x/l;
//
//	double s1 = (6*xx*xx-6*xx)/l;
//	double s2 = 1-4*xx+3*xx*xx;
//	double s3 = (6*xx-6*xx*xx)/l;
//	double s4 = -2*xx+3*xx*xx;
//
//	double ss1 = (12*xx-6)/(l*l);
//	double ss2 = (6*xx-4)/l;
//	double ss3 = (6-12*xx)/(l*l);
//	double ss4 = (6*xx-2)/l;
//
//	double f = sqrt((pow(abs((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)), 2) + pow(abs(-(p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) + (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)), 2) + pow(abs((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)), 2)));
//	double g1 = sqrt((pow(abs(p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4), 2) + pow(abs(p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4), 2) + pow(abs(p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4), 2)));
//	double g = pow(g1,3);
//	double k = f/g;
//	if(f==0) f = 1.0;
//
//	curvatureDeriv[0] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (-s1 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) + (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss1) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (s1 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss1)) - f * (0.3e1 * g1 * s1 * p[0] + 0.3e1 * g1 * s2 * p[3] + 0.3e1 * g1 * s3 * p[6] + 0.3e1 * g1 * s4 * p[9]) * s1);
//	curvatureDeriv[1] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (s1 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss1) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (-s1 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) + (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss1)) - f * (0.3e1 * g1 * s1 * p[1] + 0.3e1 * g1 * s2 * p[4] + 0.3e1 * g1 * s3 * p[7] + 0.3e1 * g1 * s4 * p[10]) * s1);
//	curvatureDeriv[2] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (-s1 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) + (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss1) + 0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (s1 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss1)) - f * (0.3e1 * g1 * s1 * p[2] + 0.3e1 * g1 * s2 * p[5] + 0.3e1 * g1 * s3 * p[8] + 0.3e1 * g1 * s4 * p[11]) * s1);
//	curvatureDeriv[3] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (-s2 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) + (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss2) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (s2 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss2)) - f * (0.3e1 * g1 * s1 * p[0] + 0.3e1 * g1 * s2 * p[3] + 0.3e1 * g1 * s3 * p[6] + 0.3e1 * g1 * s4 * p[9]) * s2);
//	curvatureDeriv[4] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (s2 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss2) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (-s2 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) + (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss2)) - f * (0.3e1 * g1 * s1 * p[1] + 0.3e1 * g1 * s2 * p[4] + 0.3e1 * g1 * s3 * p[7] + 0.3e1 * g1 * s4 * p[10]) * s2);
//	curvatureDeriv[5] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (-s2 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) + (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss2) + 0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (s2 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss2)) - f * (0.3e1 * g1 * s1 * p[2] + 0.3e1 * g1 * s2 * p[5] + 0.3e1 * g1 * s3 * p[8] + 0.3e1 * g1 * s4 * p[11]) * s2);
//	curvatureDeriv[6] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (-s3 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) + (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss3) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (s3 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss3)) - f * (0.3e1 * g1 * s1 * p[0] + 0.3e1 * g1 * s2 * p[3] + 0.3e1 * g1 * s3 * p[6] + 0.3e1 * g1 * s4 * p[9]) * s3);
//	curvatureDeriv[7] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (s3 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss3) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (-s3 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) + (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss3)) - f * (0.3e1 * g1 * s1 * p[1] + 0.3e1 * g1 * s2 * p[4] + 0.3e1 * g1 * s3 * p[7] + 0.3e1 * g1 * s4 * p[10]) * s3);
//	curvatureDeriv[8] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (-s3 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) + (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss3) + 0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (s3 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss3)) - f * (0.3e1 * g1 * s1 * p[2] + 0.3e1 * g1 * s2 * p[5] + 0.3e1 * g1 * s3 * p[8] + 0.3e1 * g1 * s4 * p[11]) * s3);
//	curvatureDeriv[9] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (-s4 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) + (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss4) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (s4 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss4)) - f * (0.3e1 * g1 * s1 * p[0] + 0.3e1 * g1 * s2 * p[3] + 0.3e1 * g1 * s3 * p[6] + 0.3e1 * g1 * s4 * p[9]) * s4);
//	curvatureDeriv[10] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (s4 * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * ss4) + 0.1e1 / f * ((p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) - (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4)) * (-s4 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) + (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss4)) - f * (0.3e1 * g1 * s1 * p[1] + 0.3e1 * g1 * s2 * p[4] + 0.3e1 * g1 * s3 * p[7] + 0.3e1 * g1 * s4 * p[10]) * s4);
//	curvatureDeriv[11] = pow(g, -0.2e1) * (g * (0.1e1 / f * ((p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4) - (p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4)) * (-s4 * (p[1] * ss1 + p[4] * ss2 + p[7] * ss3 + p[10] * ss4) + (p[1] * s1 + p[4] * s2 + p[7] * s3 + p[10] * s4) * ss4) + 0.1e1 / f * ((p[2] * s1 + p[5] * s2 + p[8] * s3 + p[11] * s4) * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * (p[2] * ss1 + p[5] * ss2 + p[8] * ss3 + p[11] * ss4)) * (s4 * (p[0] * ss1 + p[3] * ss2 + p[6] * ss3 + p[9] * ss4) - (p[0] * s1 + p[3] * s2 + p[6] * s3 + p[9] * s4) * ss4)) - f * (0.3e1 * g1 * s1 * p[2] + 0.3e1 * g1 * s2 * p[5] + 0.3e1 * g1 * s3 * p[8] + 0.3e1 * g1 * s4 * p[11]) * s4);
//
//	return k;
//}
//
//__device__ void addInternalForceAComponentFirst(int elementNum, double A, double E, double x, double l, double* p, double* intForce, double* strainD, double wtl)
//{
//	// update internal force vector (A component)
//	intForce = &intForce[12*elementNum];
//	strainD = &strainD[12*elementNum];
//	double strain = calculateStrainDerivative(&p[12*elementNum],strainD,x,l);
//	for(int i=6;i<12;i++)
//	{
//		intForce[i] = 0.5*A*E*l*strainD[i]*strain*wtl;
//	}
//}
//
//__device__ void addInternalForceAComponent(int elementNum, double A, double E, double x, double l, double* p, double* intForce, double* strainD, double wtl)
//{
//	// update internal force vector (A component)
//	intForce = &intForce[12*elementNum];
//	strainD = &strainD[12*elementNum];
//	double strain = calculateStrainDerivative(&p[12*elementNum],strainD,x,l);
//	for(int i=6;i<12;i++)
//	{
//		intForce[i] += 0.5*A*E*l*strainD[i]*strain*wtl;
//	}
//}
//
//__device__ void addInternalForceBComponent(int elementNum, double Ix, double E, double x, double l, double* p, double* intForce, double* curvatureDeriv, double wtl)
//{
//	// update internal force vector (B component)
//	intForce = &intForce[12*elementNum];
//	curvatureDeriv = &curvatureDeriv[12*elementNum];
//	double curvature = calculateCurvatureDerivative(&p[12*elementNum],curvatureDeriv,x,l);
//	for(int i=6;i<12;i++)
//	{
//		intForce[i] += 0.5*Ix*E*l*curvatureDeriv[i]*curvature*wtl;
//	}
//
//}
//
//__global__ void updateInternalForce(double* p, double* intForce, double* strainD, double* curvatureDeriv, Material* materials, int numElements)
//{
//	int i = threadIdx.x+blockIdx.x*blockDim.x;
//
//	if(i<numElements)
//	{
//		// get materials and geometry
//		Material material = materials[i];
//		double r = material.r;
//		double l = material.l;
//		double E = material.E;
//		double Ix = .25*PI*r*r*r*r;
//		double A = PI*r*r;
//
//		// determine internal force
//		double x = 0;
//		double wtl = 0;
//
//		x = -(sqrt(5.0 + 2.0*sqrt(10.0/7.0)))/3.0;
//		x = l*(1 + x)*.5;
//		wtl = (322.0 - 13.0*sqrt(70.0))/900.0;
//		addInternalForceAComponentFirst(i,A,E,x,l,p,intForce,strainD,wtl);
//
//		x = -(sqrt(5.0 - 2.0*sqrt(10.0/7.0)))/3.0;
//		x = l*(1 + x)*.5;
//		wtl = (322.0 + 13.0*sqrt(70.0))/900.0;
//		addInternalForceAComponent(i,A,E,x,l,p,intForce,strainD,wtl);
//
//		x = 0.0;
//		x = l*(1 + x)*.5;
//		wtl = 128.0/225.0;
//		addInternalForceAComponent(i,A,E,x,l,p,intForce,strainD,wtl);
//
//		x = (sqrt(5.0 - 2.0*sqrt(10.0/7.0)))/3.0;
//		x = l*(1 + x)*.5;
//		wtl = (322.0 + 13.0*sqrt(70.0))/900.0;
//		addInternalForceAComponent(i,A,E,x,l,p,intForce,strainD,wtl);
//
//		x = (sqrt(5.0+2.0*sqrt(10.0/7.0)))/3.0;
//		x = l*(1 + x)*.5;
//		wtl = (322.0 - 13.0*sqrt(70.0))/900.0;
//		addInternalForceAComponent(i,A,E,x,l,p,intForce,strainD,wtl);
//
//		x = -sqrt(3.0/5.0);
//		x = l*(1 + x)*.5;
//		wtl = 5.0/9.0;
//		addInternalForceBComponent(i,Ix,E,x,l,p,intForce,curvatureDeriv,wtl);
//
//		x = 0.0;
//		x = l*(1 + x)*.5;
//		wtl = 8.0/9.0;
//		addInternalForceBComponent(i,Ix,E,x,l,p,intForce,curvatureDeriv,wtl);
//
//		x = sqrt(3.0/5.0);
//		x = l*(1 + x)*.5;
//		wtl = 5.0/9.0;
//		addInternalForceBComponent(i,Ix,E,x,l,p,intForce,curvatureDeriv,wtl);
//	}
//}
//
//int ANCFSystem::updateInternalForces()
//{
//	// reset internal force to zero
//	//thrust::fill(fint_d.begin(),fint_d.end(),0);
//
//	// run update kernel
//	updateInternalForce<<<dimGridElement,dimBlockElement>>>(CASTD1(pnew_d),CASTD1(fint_d),CASTD1(strainDerivative_d),CASTD1(curvatureDerivative_d),CASTM1(materials_d),elements.size());
//	//cudaThreadSynchronize();
//
//	return 0;
//}
//*/
