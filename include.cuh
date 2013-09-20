#ifndef ANCFINCLUDE_H
#define ANCFINCLUDE_H

//#include <armadillo>

#include <cuda.h>
#include <helper_math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <GL/glut.h>
#include "omp.h"
#include <sscd.h>

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <cusp/array2d.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/ainv.h>
#include <cusp/transpose.h>
#include <thrust/functional.h>
#include <cusp/linear_operator.h>

using namespace std;
//using namespace arma;

// use array1d_view to wrap the individual arrays
typedef typename cusp::array1d_view<thrust::device_ptr<int> > DeviceIndexArrayView;
typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

//combine the three array1d_views into a coo_matrix_view
typedef cusp::coo_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView,
		DeviceValueArrayView> DeviceView;

#define BLOCKDIMELEMENT 512
#define BLOCKDIMCONSTRAINT 512
#define BLOCKDIMCOLLISION 512

// takes care of some GCC issues
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

//defines to cast thrust vectors as raw pointers
#define CASTC1(x) (char*)thrust::raw_pointer_cast(&x[0])
#define CASTU1(x) (uint*)thrust::raw_pointer_cast(&x[0])
#define CASTU2(x) (uint2*)thrust::raw_pointer_cast(&x[0])
#define CASTU3(x) (uint3*)thrust::raw_pointer_cast(&x[0])
#define CASTI1(x) (int*)thrust::raw_pointer_cast(&x[0])
#define CASTI2(x) (int2*)thrust::raw_pointer_cast(&x[0])
#define CASTI3(x) (int3*)thrust::raw_pointer_cast(&x[0])
#define CASTI4(x) (int4*)thrust::raw_pointer_cast(&x[0])
#define CASTF1(x) (float*)thrust::raw_pointer_cast(&x[0])
#define CASTF2(x) (float2*)thrust::raw_pointer_cast(&x[0])
#define CASTF3(x) (float3*)thrust::raw_pointer_cast(&x[0])
#define CASTF4(x) (float4*)thrust::raw_pointer_cast(&x[0])
#define CASTD1(x) (double*)thrust::raw_pointer_cast(&x[0])
#define CASTD2(x) (double2*)thrust::raw_pointer_cast(&x[0])
#define CASTD3(x) (double3*)thrust::raw_pointer_cast(&x[0])
#define CASTD4(x) (double4*)thrust::raw_pointer_cast(&x[0])
#define CASTM1(x) (Material*)thrust::raw_pointer_cast(&x[0])
#define CASTMP(x) (MaterialParticle*)thrust::raw_pointer_cast(&x[0])
#define CASTLL(x) (long long*)thrust::raw_pointer_cast(&x[0])
#define CASTCOLL(x) (collision*)thrust::raw_pointer_cast(&x[0])

//====================================THRUST=================================//
#define Thrust_Inclusive_Scan_Sum(x,y) thrust::inclusive_scan(x.begin(),x.end(), x.begin()); y=x.back();
#define Thrust_Sort_By_Key(x,y) thrust::sort_by_key(x.begin(),x.end(),y.begin())
#define Thrust_Reduce_By_KeyA(x,y,z)x= thrust::reduce_by_key(y.begin(),y.end(),thrust::constant_iterator<uint>(1),y.begin(),z.begin()).first-y.begin()
#define Thrust_Reduce_By_KeyB(x,y,z,w)x=thrust::reduce_by_key(y.begin(),y.end(),thrust::constant_iterator<uint>(1),z.begin(),w.begin()).first-z.begin()
#define Thrust_Inclusive_Scan(x) thrust::inclusive_scan(x.begin(), x.end(), x.begin())
#define Thrust_Fill(x,y) thrust::fill(x.begin(),x.end(),y)
#define Thrust_Sort(x) thrust::sort(x.begin(),x.end())
#define Thrust_Count(x,y) thrust::count(x.begin(),x.end(),y)
#define Thrust_Sequence(x) thrust::sequence(x.begin(),x.end())
#define Thrust_Equal(x,y) thrust::equal(x.begin(),x.end(), y.begin())
#define Thrust_Max(x) x[thrust::max_element(x.begin(),x.end())-x.begin()]
#define Thrust_Min(x) x[thrust::max_element(x.begin(),x.end())-x.begin()]
#define Thrust_Total(x) thrust::reduce(x.begin(),x.end())

#define PI 3.141592653589793238462643383279
#define GRAVITYx 0
#define GRAVITYy -981
#define GRAVITYz 0
#define OGL 1
#define SCALE 1

// constraint identifiers
#define CONSTRAINTABSOLUTEX 0
#define CONSTRAINTABSOLUTEY 1
#define CONSTRAINTABSOLUTEZ 2

#define CONSTRAINTABSOLUTEDX1 3
#define CONSTRAINTABSOLUTEDY1 4
#define CONSTRAINTABSOLUTEDZ1 5

#define CONSTRAINTABSOLUTEDX2 6
#define CONSTRAINTABSOLUTEDY2 7
#define CONSTRAINTABSOLUTEDZ2 8

#define CONSTRAINTRELATIVEX 9
#define CONSTRAINTRELATIVEY 10
#define CONSTRAINTRELATIVEZ 11

#define CONSTRAINTRELATIVEDX1 12
#define CONSTRAINTRELATIVEDY1 13
#define CONSTRAINTRELATIVEDZ1 14

#define CONSTRAINTRELATIVEDX2 15
#define CONSTRAINTRELATIVEDY2 16
#define CONSTRAINTRELATIVEDZ2 17

// linear operator y = A*x
class stencil: public cusp::linear_operator<double, cusp::device_memory> {
public:
	typedef cusp::linear_operator<double, cusp::device_memory> super;

	int N;
	DeviceView massMatrix;
	DeviceView stiffnessMatrix;
	DeviceValueArrayView temp;

// constructor
	stencil(int N, DeviceView mass, DeviceView stiffness,
			DeviceValueArrayView tempVector) :
			super(N, N), N(N) {
		massMatrix = mass;
		stiffnessMatrix = stiffness;
		temp = tempVector;
	}

// linear operator y = A*x
	template<typename VectorType1, typename VectorType2>
	void operator()(const VectorType1& x, VectorType2& y) const {
// obtain a raw pointer to device memory
		cusp::multiply(massMatrix, x, temp);
		cusp::multiply(stiffnessMatrix, x, y);
		cusp::blas::axpy(temp, y, 1);
	}
};

struct Material {
	double r;
	double nu;
	double E;
	double rho;
	double l;
	int numContactPoints;
};

struct MaterialParticle {
	double r;
	double nu;
	double E;
	double mass;
	double massInverse;
	int numContactPoints;
};

////////////////////////Quaternion and Vector Code////////////////////////
typedef double camreal;

struct camreal3 {

	camreal3(camreal a = 0, camreal b = 0, camreal c = 0) :
			x(a), y(b), z(c) {
	}

	camreal x, y, z;
};
struct camreal4 {

	camreal4(camreal d = 0, camreal a = 0, camreal b = 0, camreal c = 0) :
			w(d), x(a), y(b), z(c) {
	}

	camreal w, x, y, z;
};

static camreal3 operator +(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x + lhs.x;
	temp.y = rhs.y + lhs.y;
	temp.z = rhs.z + lhs.z;
	return temp;
}
static camreal3 operator -(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x - lhs.x;
	temp.y = rhs.y - lhs.y;
	temp.z = rhs.z - lhs.z;
	return temp;
}
static void operator +=(camreal3 &rhs, const camreal3 lhs) {
	rhs = rhs + lhs;
}

static void operator -=(camreal3 &rhs, const camreal3 lhs) {
	rhs = rhs - lhs;
}

static camreal3 operator *(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x * lhs.x;
	temp.y = rhs.y * lhs.y;
	temp.z = rhs.z * lhs.z;
	return temp;
}

static camreal3 operator *(const camreal3 rhs, const camreal lhs) {
	camreal3 temp;
	temp.x = rhs.x * lhs;
	temp.y = rhs.y * lhs;
	temp.z = rhs.z * lhs;
	return temp;
}

static inline camreal3 cross(camreal3 a, camreal3 b) {
	return camreal3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

static camreal4 Q_from_AngAxis(camreal angle, camreal3 axis) {
	camreal4 quat;
	camreal halfang;
	camreal sinhalf;
	halfang = (angle * 0.5);
	sinhalf = sin(halfang);
	quat.w = cos(halfang);
	quat.x = axis.x * sinhalf;
	quat.y = axis.y * sinhalf;
	quat.z = axis.z * sinhalf;
	return (quat);
}

static camreal4 normalize(const camreal4 &a) {
	camreal length = 1.0 / sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z);
	return camreal4(a.w * length, a.x * length, a.y * length, a.z * length);
}

static inline camreal4 inv(camreal4 a) {
//return (1.0f / (dot(a, a))) * F4(a.x, -a.y, -a.z, -a.w);
	camreal4 temp;
	camreal t1 = a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z;
	t1 = 1.0 / t1;
	temp.w = t1 * a.w;
	temp.x = -t1 * a.x;
	temp.y = -t1 * a.y;
	temp.z = -t1 * a.z;
	return temp;
}

static inline camreal4 mult(const camreal4 &a, const camreal4 &b) {
	camreal4 temp;
	temp.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
	temp.x = a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y;
	temp.y = a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z;
	temp.z = a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x;
	return temp;
}

static inline camreal3 quatRotate(const camreal3 &v, const camreal4 &q) {
	camreal4 r = mult(mult(q, camreal4(0, v.x, v.y, v.z)), inv(q));
	return camreal3(r.x, r.y, r.z);
}

static camreal4 operator %(const camreal4 rhs, const camreal4 lhs) {
	return mult(rhs, lhs);
}
////////////////////////END Quaternion and Vector Code////////////////////////


class OpenGLCamera {
public:
	OpenGLCamera(camreal3 pos, camreal3 lookat, camreal3 up,
			camreal viewscale) {
		max_pitch_rate = 5;
		max_heading_rate = 5;
		camera_pos = pos;
		look_at = lookat;
		camera_up = up;
		camera_heading = 0;
		camera_pitch = 0;
		dir = camreal3(0, 0, 1);
		mouse_pos = camreal3(0, 0, 0);
		camera_pos_delta = camreal3(0, 0, 0);
		scale = viewscale;
	}
	void ChangePitch(GLfloat degrees) {
		if (fabs(degrees) < fabs(max_pitch_rate)) {
			camera_pitch += degrees;
		} else {
			if (degrees < 0) {
				camera_pitch -= max_pitch_rate;
			} else {
				camera_pitch += max_pitch_rate;
			}
		}

		if (camera_pitch > 360.0f) {
			camera_pitch -= 360.0f;
		} else if (camera_pitch < -360.0f) {
			camera_pitch += 360.0f;
		}
	}
	void ChangeHeading(GLfloat degrees) {
		if (fabs(degrees) < fabs(max_heading_rate)) {
			if (camera_pitch > 90 && camera_pitch < 270
					|| (camera_pitch < -90 && camera_pitch > -270)) {
				camera_heading -= degrees;
			} else {
				camera_heading += degrees;
			}
		} else {
			if (degrees < 0) {
				if ((camera_pitch > 90 && camera_pitch < 270)
						|| (camera_pitch < -90 && camera_pitch > -270)) {
					camera_heading += max_heading_rate;
				} else {
					camera_heading -= max_heading_rate;
				}
			} else {
				if (camera_pitch > 90 && camera_pitch < 270
						|| (camera_pitch < -90 && camera_pitch > -270)) {
					camera_heading -= max_heading_rate;
				} else {
					camera_heading += max_heading_rate;
				}
			}
		}

		if (camera_heading > 360.0f) {
			camera_heading -= 360.0f;
		} else if (camera_heading < -360.0f) {
			camera_heading += 360.0f;
		}
	}
	void Move2D(int x, int y) {
		camreal3 mouse_delta = mouse_pos - camreal3(x, y, 0);
		ChangeHeading(.02 * mouse_delta.x);
		ChangePitch(.02 * mouse_delta.y);
		mouse_pos = camreal3(x, y, 0);
	}
	void SetPos(int button, int state, int x, int y) {
		mouse_pos = camreal3(x, y, 0);
	}
	void Update() {
		camreal4 pitch_quat, heading_quat;
		camreal3 angle;
		angle = cross(dir, camera_up);
		pitch_quat = Q_from_AngAxis(camera_pitch, angle);
		heading_quat = Q_from_AngAxis(camera_heading, camera_up);
		camreal4 temp = (pitch_quat % heading_quat);
		temp = normalize(temp);
		dir = quatRotate(dir, temp);
		camera_pos += camera_pos_delta;
		look_at = camera_pos + dir * 1;
		camera_heading *= .5;
		camera_pitch *= .5;
		camera_pos_delta = camera_pos_delta * .5;
		gluLookAt(camera_pos.x, camera_pos.y, camera_pos.z, look_at.x,
				look_at.y, look_at.z, camera_up.x, camera_up.y, camera_up.z);
	}
	void Forward() {
		camera_pos_delta += dir * scale;
	}
	void Back() {
		camera_pos_delta -= dir * scale;
	}
	void Right() {
		camera_pos_delta += cross(dir, camera_up) * scale;
	}
	void Left() {
		camera_pos_delta -= cross(dir, camera_up) * scale;
	}
	void Up() {
		camera_pos_delta -= camera_up * scale;
	}
	void Down() {
		camera_pos_delta += camera_up * scale;
	}

	camreal max_pitch_rate, max_heading_rate;
	camreal3 camera_pos, look_at, camera_up;
	camreal camera_heading, camera_pitch, scale;
	camreal3 dir, mouse_pos, camera_pos_delta;
};

class Node {
public:
	double x;
	double y;
	double z;
	double dx1;
	double dy1;
	double dz1;

	Node() {
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->dx1 = 1;
		this->dy1 = 0;
		this->dz1 = 0;
	}

	Node(double x, double y, double z, double dx1, double dy1, double dz1) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->dx1 = dx1;
		this->dy1 = dy1;
		this->dz1 = dz1;
	}

	Node(float3 pos, float3 dir) {
		this->x = pos.x;
		this->y = pos.y;
		this->z = pos.z;
		this->dx1 = dir.x;
		this->dy1 = dir.y;
		this->dz1 = dir.z;
	}

	double getLength(Node node1, Node node2) {
		return sqrt(
				pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2)
						+ pow(node1.z - node2.z, 2));
	}
};

class Element {
private:
	int index;
	Node node0;
	Node node1;
	double r;
	double nu;
	double E;
	double rho;
	double l;
	double collisionRadius;

public:
	Element() {
		// create test element!
		this->node0 = Node(0, 0, 0, 1, 0, 0);
		this->node1 = Node(100, 0, 0, 1, 0, 0);
		this->r = .02 * 100;
		this->nu = .3;
		this->E = 2.e7/100/100;
		this->rho = 1150.0e-6;
		this->l = 1.0;
		collisionRadius = 0;
	}
	Element(Node node0, Node node1) {
		this->node0 = node0;
		this->node1 = node1;
		this->r = .02 * 100;
		this->l = getLength(node0, node1);
		this->nu = .3;
		this->E = 2.0e5;
		this->rho = 1150.0e-6;
		collisionRadius = 0;
	}
	/*
	 Element(Node firstNode, Node lastNode, int linear)
	 {
	 this->firstNode=firstNode;
	 this->lastNode=lastNode;
	 if(linear)
	 {
	 double mag = sqrt(pow(firstNode.x-lastNode.x,2)+pow(firstNode.y-lastNode.y,2)+pow(firstNode.z-lastNode.z,2));
	 this->firstNode.dx = (lastNode.x-firstNode.x)/mag;
	 this->firstNode.dy = (lastNode.y-firstNode.y)/mag;
	 this->firstNode.dz = (lastNode.z-firstNode.z)/mag;
	 this->lastNode.dx = (lastNode.x-firstNode.x)/mag;
	 this->lastNode.dy = (lastNode.y-firstNode.y)/mag;
	 this->lastNode.dz = (lastNode.z-firstNode.z)/mag;
	 }
	 this->r=.01;
	 this->nu=.3;
	 this->E=2.0e7;
	 this->rho=7200.0;
	 this->I=PI*r*r*r*r*.25;
	 this->l=getLength(firstNode,lastNode);
	 }
	 Element(Node firstNode, Node lastNode, double r, double E, double rho, double nu)
	 {
	 this->firstNode=firstNode;
	 this->lastNode=lastNode;
	 this->r=r;
	 this->nu=nu;
	 this->E=E;
	 this->rho=rho;
	 this->I=PI*r*r*r*r*.25;
	 this->l=getLength(firstNode,lastNode);
	 }
	 */
	Node getNode0() {
		return this->node0;
	}
	Node getNode1() {
		return this->node1;
	}
	double getRadius() {
		return this->r;
	}
	double getNu() {
		return this->nu;
	}
	double getDensity() {
		return this->rho;
	}
	double getLength_l() {
		return this->l;
	}
	double getLength(Node node1, Node node2) {
		return sqrt(
				pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2)
						+ pow(node1.z - node2.z, 2));
	}
	double getElasticModulus() {
		return this->E;
	}
	int getElementIndex() {
		return this->index;
	}
	int getCollisionRadius() {
		return this->collisionRadius;
	}
	int setLength_l(double l) {
		this->l = l;
		return 0;
	}
	int setRadius(double r) {
		this->r = r;
		return 0;
	}
	int setNu(double nu) {
		this->nu = nu;
		return 0;
	}
	int setDensity(double rho) {
		this->rho = rho;
		return 0;
	}
	int setElasticModulus(double E) {
		this->E = E;
		return 0;
	}
	int setElementIndex(int index) {
		this->index = index;
		return 0;
	}
	int setCollisionRadius(double collisionRadius) {
		this->collisionRadius = collisionRadius;
		return 0;
	}
};

class Particle {
private:
	int index;
	double r;
	double nu;
	double E;
	double mass;
	float3 initialPosition;
	float3 initialVelocity;
public:
	Particle() {
		// create test element!
		this->r = .1;
		this->nu = .3;
		this->E = 2.e7;
		this->mass = 1;
		this->initialPosition = make_float3(0, 0, 0);
		this->initialVelocity = make_float3(0, 0, 0);
	}
	Particle(double r, double mass, float3 initialPosition,
			float3 initialVelocity) {
		// create test element!
		this->r = r;
		this->nu = .3;
		this->E = 2.e7;
		this->mass = mass;
		this->initialPosition = initialPosition;
		this->initialVelocity = initialVelocity;
	}
	float3 getInitialPosition() {
		return this->initialPosition;
	}
	float3 getInitialVelocity() {
		return this->initialVelocity;
	}
	double getRadius() {
		return this->r;
	}
	double getNu() {
		return this->nu;
	}
	double getMass() {
		return this->mass;
	}
	double getElasticModulus() {
		return this->E;
	}
	int getParticleIndex() {
		return this->index;
	}
	int setRadius(double r) {
		this->r = r;
		return 0;
	}
	int setNu(double nu) {
		this->nu = nu;
		return 0;
	}
	int setMass(double mass) {
		this->mass = mass;
		return 0;
	}
	int setElasticModulus(double E) {
		this->E = E;
		return 0;
	}
	int setParticleIndex(int index) {
		this->index = index;
		return 0;
	}
};

class Constraint {
public:
	int nodeNum;
	int nodeNum2;
	int constraintType;
	int2 dofLoc;

	Constraint(int nodeNum, int constraintType) {
		this->nodeNum = nodeNum;
		this->nodeNum2 = -1;
		this->constraintType = constraintType;

		switch (constraintType) {
		case CONSTRAINTABSOLUTEX:
			dofLoc.x = 6 * nodeNum;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEY:
			dofLoc.x = 6 * nodeNum + 1;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEZ:
			dofLoc.x = 6 * nodeNum + 2;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDX1:
			dofLoc.x = 6 * nodeNum + 3;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDY1:
			dofLoc.x = 6 * nodeNum + 4;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDZ1:
			dofLoc.x = 6 * nodeNum + 5;
			dofLoc.y = -1;
			break;
		}
	}
	Constraint(int nodeNum1, int nodeNum2, int constraintType) {
		this->nodeNum = nodeNum1;
		this->nodeNum2 = nodeNum2;
		this->constraintType = constraintType;

		switch (constraintType) {
		case CONSTRAINTABSOLUTEX:
			dofLoc.x = 6 * nodeNum;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEY:
			dofLoc.x = 6 * nodeNum + 1;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEZ:
			dofLoc.x = 6 * nodeNum + 2;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDX1:
			dofLoc.x = 6 * nodeNum + 3;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDY1:
			dofLoc.x = 6 * nodeNum + 4;
			dofLoc.y = -1;
			break;
		case CONSTRAINTABSOLUTEDZ1:
			dofLoc.x = 6 * nodeNum + 5;
			dofLoc.y = -1;
			break;
		case CONSTRAINTRELATIVEX:
			dofLoc.x = 6 * nodeNum;
			dofLoc.y = 6 * nodeNum2;
			break;
		case CONSTRAINTRELATIVEY:
			dofLoc.x = 6 * nodeNum + 1;
			dofLoc.y = 6 * nodeNum2 + 1;
			break;
		case CONSTRAINTRELATIVEZ:
			dofLoc.x = 6 * nodeNum + 2;
			dofLoc.y = 6 * nodeNum2 + 2;
			break;
		case CONSTRAINTRELATIVEDX1:
			dofLoc.x = 6 * nodeNum + 3;
			dofLoc.y = 6 * nodeNum2 + 3;
			break;
		case CONSTRAINTRELATIVEDY1:
			dofLoc.x = 6 * nodeNum + 4;
			dofLoc.y = 6 * nodeNum2 + 4;
			break;
		case CONSTRAINTRELATIVEDZ1:
			dofLoc.x = 6 * nodeNum + 5;
			dofLoc.y = 6 * nodeNum2 + 5;
			break;
		}
	}
};

class ANCFSystem {
public:

	ofstream posFile;
	ofstream resultsFile1;
	ofstream resultsFile2;
	ofstream resultsFile3;

	// variables
	int timeIndex;
	double time; //current time
	double simTime; //time to end simulation
	double h; //time step

	double alphaHHT;
	double betaHHT;
	double gammaHHT;
	double tol;

	// cusp
	DeviceValueArrayView eAll;
	DeviceValueArrayView eTop;
	DeviceValueArrayView eBottom;
	DeviceValueArrayView lhsVec;
	DeviceValueArrayView lhsVecStiffness;
	DeviceValueArrayView p;
	DeviceValueArrayView v;
	DeviceValueArrayView a;
	DeviceValueArrayView pnew;
	DeviceValueArrayView vnew;
	DeviceValueArrayView anewAll;
	DeviceValueArrayView anew;
	DeviceValueArrayView lambda;
	DeviceValueArrayView fext;
	DeviceValueArrayView fint;
	DeviceValueArrayView fapp;
	DeviceValueArrayView fcon;
	DeviceValueArrayView phi;
	DeviceValueArrayView phi0;
	DeviceValueArrayView phiqlam;
	DeviceValueArrayView delta;

	// cusp vectors for conjugate gradient
	DeviceValueArrayView rcg;
	DeviceValueArrayView pcg;

	// additional vectors for BiCGStab
	DeviceValueArrayView rhatcg;
	DeviceValueArrayView phatcg;
	DeviceValueArrayView residual;

	DeviceView lhs;
	DeviceView phiq;
	DeviceView mass;
	DeviceView stiffness;

	// host vectors
	thrust::host_vector<double> e_h;
	thrust::host_vector<double> p_h;
	thrust::host_vector<double> v_h;
	thrust::host_vector<double> a_h;
	thrust::host_vector<double> pnew_h;
	thrust::host_vector<double> vnew_h;
	thrust::host_vector<double> anew_h;
	thrust::host_vector<double> lhsVec_h;
	thrust::host_vector<double> lhsVecStiffness_h;
	thrust::host_vector<double> fext_h;
	thrust::host_vector<double> fint_h;
	thrust::host_vector<double> fapp_h;
	thrust::host_vector<double> fcon_h;
	thrust::host_vector<double> phi_h;
	thrust::host_vector<double> phi0_h;
	thrust::host_vector<double> phiqlam_h;
	thrust::host_vector<double> delta_h;
	thrust::host_vector<int2> constraintPairs_h;

	thrust::host_vector<int> lhsI_h;
	thrust::host_vector<int> lhsJ_h;
	thrust::host_vector<double> lhs_h;

	thrust::host_vector<int> massI_h;
	thrust::host_vector<int> massJ_h;
	thrust::host_vector<double> mass_h;

	thrust::host_vector<int> phiqI_h;
	thrust::host_vector<int> phiqJ_h;
	thrust::host_vector<double> phiq_h;

	thrust::host_vector<int> constraintsI_h;
	thrust::host_vector<int> constraintsJ_h;
	thrust::host_vector<double> constraints_h;

	thrust::host_vector<int> stiffnessI_h;
	thrust::host_vector<int> stiffnessJ_h;
	thrust::host_vector<double> stiffness_h;

	// device vectors
	thrust::device_vector<double> e_d;
	thrust::device_vector<double> p_d;
	thrust::device_vector<double> v_d;
	thrust::device_vector<double> a_d;
	thrust::device_vector<double> pnew_d;
	thrust::device_vector<double> vnew_d;
	thrust::device_vector<double> anew_d;
	thrust::device_vector<double> lhsVec_d;
	thrust::device_vector<double> lhsVecStiffness_d;
	thrust::device_vector<double> fext_d;
	thrust::device_vector<double> fint_d;
	thrust::device_vector<double> fapp_d;
	thrust::device_vector<double> fcon_d;
	thrust::device_vector<double> phi_d;
	thrust::device_vector<double> phi0_d;
	thrust::device_vector<double> phiqlam_d;
	thrust::device_vector<double> delta_d;
	thrust::device_vector<int2> constraintPairs_d;

	// cusp vectors for conjugate gradient
	thrust::device_vector<double> rcg_d;
	thrust::device_vector<double> pcg_d;
	thrust::device_vector<double> rhatcg_d;
	thrust::device_vector<double> phatcg_d;
	thrust::device_vector<double> residual_d;
	// end cusp cg stuff

	thrust::device_vector<int> lhsI_d;
	thrust::device_vector<int> lhsJ_d;
	thrust::device_vector<double> lhs_d;

	thrust::device_vector<int> massI_d;
	thrust::device_vector<int> massJ_d;
	thrust::device_vector<double> mass_d;

	thrust::device_vector<int> phiqI_d;
	thrust::device_vector<int> phiqJ_d;
	thrust::device_vector<double> phiq_d;

	thrust::device_vector<int> constraintsI_d;
	thrust::device_vector<int> constraintsJ_d;
	thrust::device_vector<double> constraints_d;

	thrust::device_vector<int> stiffnessI_d;
	thrust::device_vector<int> stiffnessJ_d;
	thrust::device_vector<double> stiffness_d;

	thrust::host_vector<double> wt5;
	thrust::host_vector<double> pt5;
	thrust::host_vector<double> wt3;
	thrust::host_vector<double> pt3;

	thrust::host_vector<double> strainDerivative_h;
	thrust::host_vector<double> strain_h;

	thrust::host_vector<double> Sx_h;
	thrust::host_vector<double> Sxx_h;

	thrust::device_vector<double> strainDerivative_d;
	thrust::device_vector<double> curvatureDerivative_d;
	thrust::device_vector<double> strain_d;

	thrust::device_vector<double> Sx_d;
	thrust::device_vector<double> Sxx_d;

	dim3 dimBlockConstraint;
	dim3 dimGridConstraint;

	dim3 dimBlockElement;
	dim3 dimGridElement;

	dim3 dimBlockParticles;
	dim3 dimGridParticles;

	dim3 dimBlockCollision;
	dim3 dimGridCollision;

	//particle stuff
	thrust::host_vector<double> pParticle_h;
	thrust::host_vector<double> vParticle_h;
	thrust::host_vector<double> aParticle_h;
	thrust::host_vector<double> fParticle_h;

	thrust::device_vector<double> pParticle_d;
	thrust::device_vector<double> vParticle_d;
	thrust::device_vector<double> aParticle_d;
	thrust::device_vector<double> fParticle_d;

	CollisionDetector detector;
	thrust::host_vector<float3> aabb_data_h;
	thrust::device_vector<float3> aabb_data_d;
	//thrust::host_vector<float3> aabbMax;
	//thrust::host_vector<float3> aabbMin;
	//thrust::host_vector<uint2> aabbTypes; //(type (0 = beam, 1 = particle),index)

	thrust::host_vector<long long> potentialCollisions_h;

	thrust::host_vector<uint> collisionCounts_h;
	thrust::device_vector<uint> collisionCounts_d;
	uint numActualCollisions;
	thrust::host_vector<float3> collisionNormals_h;
	thrust::device_vector<float3> collisionNormals_d;
	thrust::host_vector<double> collisionPenetrations_h;
	thrust::device_vector<double> collisionPenetrations_d;
	thrust::host_vector<double> collisionAlongBeam_h;
	thrust::device_vector<double> collisionAlongBeam_d;
	thrust::host_vector<uint> collisionIndices1_h;
	thrust::device_vector<uint> collisionIndices1_d;
	thrust::host_vector<uint> collisionIndices2_h;
	thrust::device_vector<uint> collisionIndices2_d;

public:
	ANCFSystem();
	vector<Element> elements;
	vector<Constraint> constraints;
	thrust::host_vector<Material> materials;
	thrust::device_vector<Material> materials_d;

	vector<Particle> particles;
	thrust::host_vector<MaterialParticle> pMaterials_h;
	thrust::device_vector<MaterialParticle> pMaterials_d;

	int numContactPoints;
	int numCollisions;
	int numCollisionsSphere;
	double coefRestitution;
	double frictionCoef;
	int fileIndex;
	double timeToSimulate;

	double getCurrentTime();
	double getSimulationTime();
	double getTimeStep();
	double getTolerance();
	int getTimeIndex();
	int setSimulationTime(double simTime);
	int setTimeStep(double h);
	int setTolerance(double tolerance);
	int addElement(Element* element);
	int addParticle(Particle* particle);
	int updateParticleDynamics();
	int addForce(Element* element, double xi, float3 force);
	int clearAppliedForces();
	int getLeftHandSide(DeviceValueArrayView x);
	int DoTimeStep();
	int solve_cg();
	int solve_bicgstab();
	float3 getXYZPosition(int elementIndex, double xi);
	float3 getXYZVelocity(int elementIndex, double xi);
	float3 getXYZPositionParticle(int index);
	float3 getXYZVelocityParticle(int index);
	int calculateInitialPhi();
	int createMass();
	int initializeSystem();
	int initializeDevice();
	int updateInternalForces();
	int updateInternalForcesCPU();
	int updateInternalForcesARMA();
	int updatePhiq();
	int updatePhi();
	int writeToFile();
	int saveLHS();

//	Node getFirstNode(Element element)
//	{
//		double* ptr = p.memptr();
//		ptr = &ptr[element.getElementIndex()*12];
//		return Node(ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5]);
//	}
//
//	Node getLastNode(Element element)
//	{
//		double* ptr = p.memptr();
//		ptr = &ptr[element.getElementIndex()*12+6];
//		return Node(ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5]);
//	}

	// constraint code (by node number)
	int addConstraint_AbsoluteX(int nodeNum);
	int addConstraint_AbsoluteY(int nodeNum);
	int addConstraint_AbsoluteZ(int nodeNum);

	int addConstraint_AbsoluteDX1(int nodeNum);
	int addConstraint_AbsoluteDY1(int nodeNum);
	int addConstraint_AbsoluteDZ1(int nodeNum);

	int addConstraint_RelativeX(int nodeNum1, int nodeNum2);
	int addConstraint_RelativeY(int nodeNum1, int nodeNum2);
	int addConstraint_RelativeZ(int nodeNum1, int nodeNum2);

	int addConstraint_RelativeDX1(int nodeNum1, int nodeNum2);
	int addConstraint_RelativeDY1(int nodeNum1, int nodeNum2);
	int addConstraint_RelativeDZ1(int nodeNum1, int nodeNum2);

	int addConstraint_AbsoluteFixed(int nodeNum);
	int addConstraint_RelativeFixed(int nodeNum1, int nodeNum2);
	int addConstraint_AbsoluteSpherical(int nodeNum);
	int addConstraint_RelativeSpherical(int nodeNum1, int nodeNum2);

	// constraint code (by element)
	int addConstraint_AbsoluteX(Element& element, int node_local);
	int addConstraint_AbsoluteY(Element& element, int node_local);
	int addConstraint_AbsoluteZ(Element& element, int node_local);

	int addConstraint_AbsoluteDX1(Element& element, int node_local);
	int addConstraint_AbsoluteDY1(Element& element, int node_local);
	int addConstraint_AbsoluteDZ1(Element& element, int node_local);

	int addConstraint_RelativeX(Element& element1, int node_local1,
			Element& element2, int node_local2);
	int addConstraint_RelativeY(Element& element1, int node_local1,
			Element& element2, int node_local2);
	int addConstraint_RelativeZ(Element& element1, int node_local1,
			Element& element2, int node_local2);

	int addConstraint_RelativeDX1(Element& element1, int node_local1,
			Element& element2, int node_local2);
	int addConstraint_RelativeDY1(Element& element1, int node_local1,
			Element& element2, int node_local2);
	int addConstraint_RelativeDZ1(Element& element1, int node_local1,
			Element& element2, int node_local2);

	int addConstraint_AbsoluteFixed(Element& element, int node_local);
	int addConstraint_AbsoluteSpherical(Element& element, int node_local);
	int addConstraint_RelativeFixed(Element& element1, int node_local1,
			Element& element2, int node_local2);
	int addConstraint_RelativeSpherical(Element& element1, int node_local1,
			Element& element2, int node_local2);

	int updateBoundingBoxes_CPU();
//	int updateBoundingBoxes();
	int initializeBoundingBoxes_CPU();
//	int detectGroundContact_CPU();
//	int applyGroundContactForce_CPU(int elementIndex, double xi, double penetration);
//	int generateAllPossibleContacts();
	int detectCollisions_CPU();
	int performNarrowphaseCollisionDetection_CPU(long long potentialCollision);
	int applyContactForce_CPU(int beamIndex, int particleIndex, double penetration, double xi, float3 normal);
	int applyContactForceParticles_CPU(int particleIndex1, int particleIndex2, double penetration, float3 normal);
	int applyForce_CPU(int elementIndex, double l, double xi, float3 force);
	int applyForceParticle_CPU(int particleIndex, float3 force);
	int performNarrowphaseCollisionDetection();

	int countActualCollisions();
	int populateCollisions();
	int accumulateContactForces(int numBodiesInContact);
	int accumulateContactForces_CPU();
};

#endif
