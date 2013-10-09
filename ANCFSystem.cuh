/*
 * ANCFSystem.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef ANCFSYSTEM_CUH_
#define ANCFSYSTEM_CUH_

#include "include.cuh"
#include "Element.cuh"
#include "Constraint.cuh"
#include "Node.cuh"
#include "Particle.cuh"

#include <spike/solver.h>
#include <spike/spmv.h>

typedef typename spike::Solver<DeviceView, DeviceValueArrayView> SpikeSolver;
typedef typename spike::SpmvCusp<DeviceView, DeviceValueArrayView> SpmvFunctor;

#define GRAVITYx 0
#define GRAVITYy -9.81
#define GRAVITYz 0

class ANCFSystem {
public:

	// spike stuff
	int partitions;
	SpikeSolver* mySolver;
	SpmvFunctor* mySpmv;
	bool useSpike;
	// end spike stuff

	ofstream posFile;
	ofstream resultsFile1;
	ofstream resultsFile2;
	ofstream resultsFile3;

	// variables
	int timeIndex;
	double time; //current time
	double simTime; //time to end simulation
	double h; //time step
	bool fullJacobian;

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
	int setPartitions(int partitions);
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

#endif /* ANCFSYSTEM_CUH_ */
