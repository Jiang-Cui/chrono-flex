#include <algorithm>
#include <vector>
#include "include.cuh"
#include "ANCFSystem.cuh"

#include <cusp/io/matrix_market.h>

//// linear operator y = A*x (for CUSP)
//class stencil: public cusp::linear_operator<double, cusp::device_memory> {
//public:
//        typedef cusp::linear_operator<double, cusp::device_memory> super;
//
//        int N;
//        DeviceView massMatrix;
//        DeviceView phiqMatrix;
//        DeviceValueArrayView temp;
//
//// constructor
//        stencil(int N, DeviceView lhs_mass, DeviceView lhs_phiq,
//                        DeviceValueArrayView tempVector) :
//                        super(N, N), N(N) {
//                massMatrix = lhs_mass;
//                phiqMatrix = lhs_phiq;
//                temp = tempVector;
//        }
//
//// linear operator y = A*x
//        template<typename VectorType1, typename VectorType2>
//        void operator()(const VectorType1& x, VectorType2& y) const {
//// obtain a raw pointer to device memory
//                cusp::multiply(massMatrix, x, temp);
//                cusp::multiply(phiqMatrix, x, y);
//                cusp::blas::axpy(temp, y, 1);
//        }
//};

ANCFSystem::ANCFSystem()
{
  // Set default solver parameters
  setAlpha_HHT(-0.1);
  setTimeStep(1e-3);
  maxNewtonIterations = 20;
  gravity = make_double3(0,-9.81,0);
  deviceIndex = 0;
  cudaSetDevice(deviceIndex);

  // spike stuff
  partitions = 1;
  solverOptions.safeFactorization = true;
  solverOptions.trackReordering = true;
  solverOptions.maxNumIterations = 5000;
  preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
  preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
  // end spike stuff

  this->timeIndex = 0;
  this->time = 0;
  timeToSimulate = 0;
  simTime = 0;
  fullJacobian = 1;

  wt3.push_back(5.0 / 9.0);
  wt3.push_back(8.0 / 9.0);
  wt3.push_back(5.0 / 9.0);

  pt3.push_back(-sqrt(3.0 / 5.0));
  pt3.push_back(0.0);
  pt3.push_back(sqrt(3.0 / 5.0));

  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back(128. / 225.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);

  pt5.push_back(-(sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(-(sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(0.);
  pt5.push_back((sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back((sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);

  numCollisions = 0;
  numCollisionsSphere = 0;
  numContactPoints = 5;
  coefRestitution = .3;
  frictionCoef = .3;
  fileIndex = 0;

  // set up position files
  char filename1[100];
  char filename2[100];
  char filename3[100];
  sprintf(filename1, "position.dat");
  resultsFile1.open(filename1);
  sprintf(filename2, "energy.dat");
  resultsFile2.open(filename2);
  sprintf(filename3, "reactions.dat");
  resultsFile3.open(filename3);
}

ANCFSystem::ANCFSystem(int deviceIndex)
{
  // Set default solver parameters
  setAlpha_HHT(-0.1);
  setTimeStep(1e-3);
  maxNewtonIterations = 20;
  gravity = make_double3(0,-9.81,0);
  this->deviceIndex = deviceIndex;
  cudaSetDevice(deviceIndex);

  // spike stuff
  partitions = 1;
  solverOptions.safeFactorization = true;
  solverOptions.trackReordering = true;
  solverOptions.maxNumIterations = 50;
  preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
  preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
  // end spike stuff

  this->timeIndex = 0;
  this->time = 0;
  timeToSimulate = 0;
  simTime = 0;
  fullJacobian = 1;

  wt3.push_back(5.0 / 9.0);
  wt3.push_back(8.0 / 9.0);
  wt3.push_back(5.0 / 9.0);

  pt3.push_back(-sqrt(3.0 / 5.0));
  pt3.push_back(0.0);
  pt3.push_back(sqrt(3.0 / 5.0));

  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back(128. / 225.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);

  pt5.push_back(-(sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(-(sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(0.);
  pt5.push_back((sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back((sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);

  numCollisions = 0;
  numCollisionsSphere = 0;
  numContactPoints = 5;
  coefRestitution = .3;
  frictionCoef = .3;
  fileIndex = 0;

  // set up position files
  char filename1[100];
  char filename2[100];
  char filename3[100];
  sprintf(filename1, "position.dat");
  resultsFile1.open(filename1);
  sprintf(filename2, "energy.dat");
  resultsFile2.open(filename2);
  sprintf(filename3, "reactions.dat");
  resultsFile3.open(filename3);
}

void ANCFSystem::setSolverType(int solverType)
{
  cudaSetDevice(deviceIndex);
  switch(solverType) {
  case 0:
    solverOptions.solverType = spike::BiCGStab;
    break;
  case 1:
    solverOptions.solverType = spike::BiCGStab1;
    break;
  case 2:
    solverOptions.solverType = spike::BiCGStab2;
    break;
  case 3:
    solverOptions.solverType = spike::MINRES;
    break;
  }
}

void ANCFSystem::setPrecondType(int useSpike)
{
  cudaSetDevice(deviceIndex);
  solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void ANCFSystem::setAlpha_HHT(double alpha) {
  cudaSetDevice(deviceIndex);

  // should be greater than -.3, usually set to -.1
  alphaHHT = alpha;
  betaHHT = (1 - alphaHHT) * (1 - alphaHHT) * .25;
  gammaHHT = 0.5 - alphaHHT;
}

void ANCFSystem::setTimeStep(double step_size,
                             double precision)
{
  cudaSetDevice(deviceIndex);

  h = step_size;

  // Set tolerance for Newton iteration based on the precision in positions
  // and integration step-size.
  double safety = 1;////0.5;
  tol = safety * precision / (h * h);

  // Set the tolerances for Krylov
  solverOptions.relTol = std::min(0.01 * tol, 1e-6);
  solverOptions.absTol = 1e-10;
}

void ANCFSystem::printSolverParams()
{
  printf("Step size: %e\n", h);
  printf("Newton tolerance: %e\n", tol);
  printf("Max. Newton iterations: %d\n", maxNewtonIterations);
  printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
  printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
  printf("----------------------------\n");
}


int ANCFSystem::addParticle(Particle* particle) {
  cudaSetDevice(deviceIndex);

  //add the element
  particle->setParticleIndex(particles.size());
  this->particles.push_back(*particle);

  MaterialParticle material;
  material.E = particle->getElasticModulus();
  material.nu = particle->getNu();
  material.mass = particle->getMass();
  material.massInverse = 1.0 / particle->getMass();
  material.r = particle->getRadius();
  material.numContactPoints = 1;
  this->pMaterials_h.push_back(material);

  // update p
  float3 pos0 = particle->getInitialPosition();
  pParticle_h.push_back(pos0.x);
  pParticle_h.push_back(pos0.y);
  pParticle_h.push_back(pos0.z);

  // update v
  float3 vel0 = particle->getInitialVelocity();
  vParticle_h.push_back(vel0.x);
  vParticle_h.push_back(vel0.y);
  vParticle_h.push_back(vel0.z);

  for (int i = 0; i < 3; i++) {
    aParticle_h.push_back(0.0);
    fParticle_h.push_back(0.0);
  }

  return particles.size();
}

int ANCFSystem::addElement(Element* element) {
  cudaSetDevice(deviceIndex);

  //add the element
  element->setElementIndex(elements.size());
  this->elements.push_back(*element);

  Material material;
  material.E = element->getElasticModulus();
  material.l = element->getLength_l();
  material.nu = element->getNu();
  material.rho = element->getDensity();
  material.r = element->getRadius();
  material.numContactPoints = numContactPoints;
  this->materials.push_back(material);

  // update p
  Node node = element->getNode0();
  p_h.push_back(node.x);
  p_h.push_back(node.y);
  p_h.push_back(node.z);
  p_h.push_back(node.dx1);
  p_h.push_back(node.dy1);
  p_h.push_back(node.dz1);
  node = element->getNode1();
  p_h.push_back(node.x);
  p_h.push_back(node.y);
  p_h.push_back(node.z);
  p_h.push_back(node.dx1);
  p_h.push_back(node.dy1);
  p_h.push_back(node.dz1);

  for (int i = 0; i < 12; i++) {
    e_h.push_back(0.0);
    v_h.push_back(0.0);
    a_h.push_back(0.0);
    anew_h.push_back(0.0);
    fint_h.push_back(0.0);
    fcon_h.push_back(0.0);
    fapp_h.push_back(0.0);
    phiqlam_h.push_back(0.0);
    delta_h.push_back(0.0);
    strainDerivative_h.push_back(0.0);
  }
  strain_h.push_back(0.0);

  for (int i = 0; i < 4; i++) {
    Sx_h.push_back(0.0);
    Sxx_h.push_back(0.0);
  }

  //update other vectors (no initial velocity or acceleration)
  double r = element->getRadius();
  double a = element->getLength_l();
  double rho = element->getDensity();
  double A = PI * r * r;

  // update external force vector (gravity)
  fext_h.push_back(rho * A * a * gravity.x / 0.2e1);
  fext_h.push_back(rho * A * a * gravity.y / 0.2e1);
  fext_h.push_back(rho * A * a * gravity.z / 0.2e1);
  fext_h.push_back(rho * A * a * a * gravity.x / 0.12e2);
  fext_h.push_back(rho * A * a * a * gravity.y / 0.12e2);
  fext_h.push_back(rho * A * a * a * gravity.z / 0.12e2);
  fext_h.push_back(rho * A * a * gravity.x / 0.2e1);
  fext_h.push_back(rho * A * a * gravity.y / 0.2e1);
  fext_h.push_back(rho * A * a * gravity.z / 0.2e1);
  fext_h.push_back(-rho * A * a * a * gravity.x / 0.12e2);
  fext_h.push_back(-rho * A * a * a * gravity.y / 0.12e2);
  fext_h.push_back(-rho * A * a * a * gravity.z / 0.12e2);

  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 12; j++) {
      lhsI_h.push_back(i + 12 * (elements.size() - 1));
      lhsJ_h.push_back(j + 12 * (elements.size() - 1));
      lhs_h.push_back(0.0);
    }
  }

  return elements.size();
}

int ANCFSystem::addForce(Element* element, double xi, float3 force) {
  cudaSetDevice(deviceIndex);

  int index = element->getElementIndex();
  int l = element->getLength_l();

  //fapp_h = fapp_d;

  fapp_h[0 + 12 * index] += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.x;
  fapp_h[1 + 12 * index] += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.y;
  fapp_h[2 + 12 * index] += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.z;
  fapp_h[3 + 12 * index] += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.x;
  fapp_h[4 + 12 * index] += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.y;
  fapp_h[5 + 12 * index] += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.z;
  fapp_h[6 + 12 * index] += (3 * xi * xi - 2 * pow(xi, 3)) * force.x;
  fapp_h[7 + 12 * index] += (3 * xi * xi - 2 * pow(xi, 3)) * force.y;
  fapp_h[8 + 12 * index] += (3 * xi * xi - 2 * pow(xi, 3)) * force.z;
  fapp_h[9 + 12 * index] += l * (-xi * xi + pow(xi, 3)) * force.x;
  fapp_h[10 + 12 * index] += l * (-xi * xi + pow(xi, 3)) * force.y;
  fapp_h[11 + 12 * index] += l * (-xi * xi + pow(xi, 3)) * force.z;

  fapp_d = fapp_h;

  return 0;
}

int ANCFSystem::clearAppliedForces() {
  cudaSetDevice(deviceIndex);

  thrust::fill(fapp_d.begin(), fapp_d.end(), 0.0); //Clear internal forces
  return 0;
}

int ANCFSystem::updatePhiq() // used in Newton iteration, nice to keep it separate (but not memory efficient) - only needs to be done once (linear constraints)
{
  cudaSetDevice(deviceIndex);

  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];

    phiqJ_h.push_back(i);
    phiqI_h.push_back(constraint.dofLoc.x);
    phiq_h.push_back(1.0);

    if (constraint.nodeNum2 != -1) {
      phiqJ_h.push_back(i);
      phiqI_h.push_back(constraint.dofLoc.y);
      phiq_h.push_back(-1.0);
    }
  }
  phiqI_d = phiqI_h;
  phiqJ_d = phiqJ_h;
  phiq_d = phiq_h;

  thrust::device_ptr<int> wrapped_device_I(CASTI1(phiqI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I,
      wrapped_device_I + phiqI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(phiqJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J,
      wrapped_device_J + phiqJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(phiq_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V,
      wrapped_device_V + phiq_d.size());

  phiq = DeviceView(12 * elements.size(), constraints.size(), phiq_d.size(),
      row_indices, column_indices, values);
  phiq.sort_by_row();

  return 0;
}

__global__ void calculateRHSlower(double* phi, double* p, double* phi0,
    double factor, int2* constraintPairs, int numConstraints) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < numConstraints) {
    int2 constraintPair = constraintPairs[i];
    if (constraintPair.y == -1) {
      phi[i] = factor * (p[constraintPair.x] - phi0[i]);
    } else {
      phi[i] = factor
          * (p[constraintPair.x] - p[constraintPair.y] - phi0[i]);
    }
    __syncthreads();
  }
}

int ANCFSystem::updatePhi() {
  cudaSetDevice(deviceIndex);
  calculateRHSlower<<<dimGridConstraint,dimBlockConstraint>>>(CASTD1(phi_d), CASTD1(pnew_d), CASTD1(phi0_d), 1.0/(betaHHT*h*h), CASTI2(constraintPairs_d), constraints.size());

  return 0;
}

__global__ void updateParticleDynamics_GPU(double h, double* a, double* v,
    double* p, double* f, MaterialParticle* materials, double3 gravity, int numParticles) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < numParticles) {
    a = &a[3 * i];
    v = &v[3 * i];
    p = &p[3 * i];
    f = &f[3 * i];
    MaterialParticle material = materials[i];

    a[0] = material.massInverse * f[0] + gravity.x;
    a[1] = material.massInverse * f[1] + gravity.y;
    a[2] = material.massInverse * f[2] + gravity.z;

    for (int j = 0; j < 3; j++) {
      v[j] += h * a[j];
      p[j] += h * v[j];
    }
  }
}

int ANCFSystem::updateParticleDynamics() {
  cudaSetDevice(deviceIndex);
  updateParticleDynamics_GPU<<<dimGridParticles,dimBlockParticles>>>(h,CASTD1(aParticle_d), CASTD1(vParticle_d), CASTD1(pParticle_d), CASTD1(fParticle_d), CASTMP(pMaterials_d), gravity, particles.size());

  return 0;
}

int ANCFSystem::calculateInitialPhi() {
  cudaSetDevice(deviceIndex);
  for (int i = 0; i < constraints.size(); i++)
    phi0_h.push_back(0);
  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];

    if (constraint.nodeNum2 == -1) {
      phi0_h[i] = p_h[constraint.dofLoc.x];
    } else {
      phi0_h[i] = p_h[constraint.dofLoc.x] - p_h[constraint.dofLoc.y];
    }
  }

  return 0;
}

int ANCFSystem::initializeDevice() {
  cudaSetDevice(deviceIndex);

  pMaterials_d = pMaterials_h;
  pParticle_d = pParticle_h;
  vParticle_d = vParticle_h;
  aParticle_d = aParticle_h;
  fParticle_d = fParticle_h;

  materials_d = materials;
  strainDerivative_d = strainDerivative_h;
  curvatureDerivative_d = strainDerivative_h;
  strain_d = strain_h;
  Sx_d = Sx_h;
  Sxx_d = Sxx_h;

  e_d = e_h;
  p_d = p_h;
  v_d = v_h;
  a_d = a_h;
  pnew_d = p_h;
  vnew_d = v_h;
  anew_d = anew_h;
  fext_d = fext_h;
  fint_d = fint_h;
  fapp_d = fapp_h;
  fcon_d = fcon_h;
  phi_d = phi_h;
  phi0_d = phi0_h;
  phiqlam_d = phiqlam_h;
  delta_d = delta_h;
  constraintPairs_d = constraintPairs_h;
  lhsVec_d = anew_h;

  lhsI_d = lhsI_h;
  lhsJ_d = lhsJ_h;
  lhs_d = lhs_h;

  constraintsI_d = constraintsI_h;
  constraintsJ_d = constraintsJ_h;
  constraints_d = constraints_h;

  thrust::device_ptr<double> wrapped_device_e(CASTD1(e_d));
  thrust::device_ptr<double> wrapped_device_p(CASTD1(p_d));
  thrust::device_ptr<double> wrapped_device_v(CASTD1(v_d));
  thrust::device_ptr<double> wrapped_device_a(CASTD1(a_d));
  thrust::device_ptr<double> wrapped_device_pnew(CASTD1(pnew_d));
  thrust::device_ptr<double> wrapped_device_vnew(CASTD1(vnew_d));
  thrust::device_ptr<double> wrapped_device_anew(CASTD1(anew_d));
  thrust::device_ptr<double> wrapped_device_fext(CASTD1(fext_d));
  thrust::device_ptr<double> wrapped_device_fint(CASTD1(fint_d));
  thrust::device_ptr<double> wrapped_device_fapp(CASTD1(fapp_d));
  thrust::device_ptr<double> wrapped_device_fcon(CASTD1(fcon_d));
  thrust::device_ptr<double> wrapped_device_phi(CASTD1(phi_d));
  thrust::device_ptr<double> wrapped_device_phi0(CASTD1(phi0_d));
  thrust::device_ptr<double> wrapped_device_phiqlam(CASTD1(phiqlam_d));
  thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
  thrust::device_ptr<double> wrapped_device_lhsVec(CASTD1(lhsVec_d));

  eAll = DeviceValueArrayView(wrapped_device_e,
      wrapped_device_e + e_d.size());
  eTop = DeviceValueArrayView(wrapped_device_e,
      wrapped_device_e + 12 * elements.size());
  eBottom = DeviceValueArrayView(wrapped_device_e + 12 * elements.size(),
      wrapped_device_e + e_d.size());
  p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
  v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
  a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
  pnew = DeviceValueArrayView(wrapped_device_pnew,
      wrapped_device_pnew + pnew_d.size());
  vnew = DeviceValueArrayView(wrapped_device_vnew,
      wrapped_device_vnew + vnew_d.size());
  anewAll = DeviceValueArrayView(wrapped_device_anew,
      wrapped_device_anew + anew_d.size());
  anew = DeviceValueArrayView(wrapped_device_anew,
      wrapped_device_anew + 12 * elements.size());
  lambda = DeviceValueArrayView(wrapped_device_anew + 12 * elements.size(),
      wrapped_device_anew + anew_d.size());
  fext = DeviceValueArrayView(wrapped_device_fext,
      wrapped_device_fext + fext_d.size());
  fint = DeviceValueArrayView(wrapped_device_fint,
      wrapped_device_fint + fint_d.size());
  fapp = DeviceValueArrayView(wrapped_device_fapp,
      wrapped_device_fapp + fapp_d.size());
  fcon = DeviceValueArrayView(wrapped_device_fcon,
      wrapped_device_fcon + fcon_d.size());
  phi = DeviceValueArrayView(wrapped_device_phi,
      wrapped_device_phi + phi_d.size());
  phi0 = DeviceValueArrayView(wrapped_device_phi0,
      wrapped_device_phi0 + phi0_d.size());
  phiqlam = DeviceValueArrayView(wrapped_device_phiqlam,
      wrapped_device_phiqlam + phiqlam_d.size());
  delta = DeviceValueArrayView(wrapped_device_delta,
      wrapped_device_delta + delta_d.size());
  lhsVec = DeviceValueArrayView(wrapped_device_lhsVec,
      wrapped_device_lhsVec + lhsVec_d.size());



  // create lhs matrix using cusp library (shouldn't change)
  thrust::device_ptr<int> wrapped_device_I(CASTI1(lhsI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I,
      wrapped_device_I + lhsI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(lhsJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J,
      wrapped_device_J + lhsJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(lhs_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V,
      wrapped_device_V + lhs_d.size());

  lhs = DeviceView(anew_d.size(), anew_d.size(), lhs_d.size(), row_indices,
      column_indices, values);
  // end create lhs matrix

  // create the view to the mass block of the lhs matrix
  DeviceIndexArrayView row_indices_mass = DeviceIndexArrayView(
      wrapped_device_I, wrapped_device_I + 12 * 12 * elements.size());
  DeviceIndexArrayView column_indices_mass = DeviceIndexArrayView(
      wrapped_device_J, wrapped_device_J + 12 * 12 * elements.size());
  DeviceValueArrayView values_mass = DeviceValueArrayView(wrapped_device_V,
      wrapped_device_V + 12 * 12 * elements.size());
  lhs_mass = DeviceView(anew_d.size(), anew_d.size(),
      12 * 12 * elements.size(), row_indices_mass, column_indices_mass,
      values_mass);
  // end create the view to the mass block of the lhs matrix

  // create the view to the mass block of the lhs matrix
  DeviceIndexArrayView row_indices_phiq = DeviceIndexArrayView(
      wrapped_device_I + 12 * 12 * elements.size(),
      wrapped_device_I + lhsI_d.size());
  DeviceIndexArrayView column_indices_phiq = DeviceIndexArrayView(
      wrapped_device_J + 12 * 12 * elements.size(),
      wrapped_device_J + lhsJ_d.size());
  DeviceValueArrayView values_phiq = DeviceValueArrayView(
      wrapped_device_V + 12 * 12 * elements.size(),
      wrapped_device_V + lhs_d.size());
  lhs_phiq = DeviceView(anew_d.size(), anew_d.size(),
      lhs_d.size() - 12 * 12 * elements.size(), row_indices_phiq,
      column_indices_phiq, values_phiq);
  lhs_phiq.sort_by_row(); // MUST BE SORTED FOR SPMV TO WORK CORRECTLY
  // end create the view to the mass block of the lhs matrix

  dimBlockConstraint.x = BLOCKDIMCONSTRAINT;
  dimGridConstraint.x = static_cast<int>(ceil(
      (static_cast<double>(constraints.size()))
          / (static_cast<double>(BLOCKDIMCONSTRAINT))));

  dimBlockElement.x = BLOCKDIMELEMENT;
  dimGridElement.x = (int) ceil(
      ((double) (elements.size())) / ((double) BLOCKDIMELEMENT));

  dimBlockParticles.x = BLOCKDIMELEMENT;
  dimGridParticles.x = (int) ceil(
      ((double) (particles.size())) / ((double) BLOCKDIMELEMENT));

  dimBlockCollision.x = BLOCKDIMCOLLISION;
  dimGridCollision.x = (int) ceil(
      ((double) (particles.size())) / ((double) BLOCKDIMCOLLISION));

  return 0;
}

int ANCFSystem::initializeSystem() {
  cudaSetDevice(deviceIndex);

  ANCFSystem::updatePhiq();
  ANCFSystem::calculateInitialPhi();

  for (int i = 0; i < constraints.size(); i++) {
    delta_h.push_back(0);
    e_h.push_back(0);
    anew_h.push_back(0);
    phi_h.push_back(0);
    constraintPairs_h.push_back(constraints[i].dofLoc);
  }

  // join phi_q to lhs
  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];
    lhsI_h.push_back(i + 12 * elements.size());
    lhsJ_h.push_back(constraint.dofLoc.x);
    lhs_h.push_back(1.0);

    if (constraint.nodeNum2 != -1) {
      lhsI_h.push_back(i + 12 * elements.size());
      lhsJ_h.push_back(constraint.dofLoc.y);
      lhs_h.push_back(-1.0);
    }
  }

  // join phi_q' to lhs
  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];
    lhsJ_h.push_back(i + 12 * elements.size());
    lhsI_h.push_back(constraint.dofLoc.x);
    lhs_h.push_back(1.0);

    if (constraint.nodeNum2 != -1) {
      lhsJ_h.push_back(i + 12 * elements.size());
      lhsI_h.push_back(constraint.dofLoc.y);
      lhs_h.push_back(-1.0);
    }
  }

  // Get constraints
  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];
    constraintsI_h.push_back(i + 12 * elements.size());
    constraintsJ_h.push_back(constraint.dofLoc.x);
    constraints_h.push_back(1.0);

    if (constraint.nodeNum2 != -1) {
      constraintsI_h.push_back(i + 12 * elements.size());
      constraintsJ_h.push_back(constraint.dofLoc.y);
      constraints_h.push_back(-1.0);
    }
  }

  // join phi_q' to lhs
  for (int i = 0; i < constraints.size(); i++) {
    Constraint constraint = constraints[i];
    constraintsJ_h.push_back(i + 12 * elements.size());
    constraintsI_h.push_back(constraint.dofLoc.x);
    constraints_h.push_back(1.0);

    if (constraint.nodeNum2 != -1) {
      constraintsJ_h.push_back(i + 12 * elements.size());
      constraintsI_h.push_back(constraint.dofLoc.y);
      constraints_h.push_back(-1.0);
    }
  }

  initializeDevice();

  ANCFSystem::resetLeftHandSideMatrix();
  ANCFSystem::updateInternalForces();

  //cusp::blas::axpy(fint,eTop,-1);
  cusp::blas::axpby(fext, fint, eTop, 1, -1);

  // create and setup the Spike::GPU solver
  m_spmv = new MySpmv(lhs_mass, lhs_phiq, lhsVec);
  mySolver = new SpikeSolver(partitions, solverOptions);
  mySolver->setup(lhs);

//  char filename[100];
//  sprintf(filename, "./lhs.txt");
//  cusp::io::write_matrix_market_file(lhs, filename);
  cusp::blas::fill(anewAll, 0);
  bool success = mySolver->solve(*m_spmv, eAll, anewAll);
  spike::Stats stats = mySolver->getStats();
//  cout << "Success: " << success << " Iterations: "
//      << stats.numIterations << " relResidualNorm: "
//      << stats.relResidualNorm << endl;

  cout << endl
       << "Linear problem size:  " << eAll.size() << endl
       << "Number partitions:    " << stats.numPartitions << endl
       << "Bandwidth after RCM:  " << stats.bandwidthReorder << endl
       << "Bandwidth final:      " << stats.bandwidth << endl
       << "nuKf factor:          " << stats.nuKf << endl << endl;

  cusp::copy(anew, a);
  cusp::copy(v, vnew);
  cusp::copy(p, pnew);

  // Vectors for Spike solver stats
  spikeSolveTime.resize(maxNewtonIterations);
  spikeNumIter.resize(maxNewtonIterations);

  return 0;
}

int ANCFSystem::transferState(ANCFSystem* dst) {
  dst->pnew_d = p_d;
  //dst->p_d = p_d;
  dst->vnew_d = v_d;
  //dst->v_d = v_d;
  dst->anew_d = a_d;
  //dst->a_d = a_d;
  dst->time = time;
  dst->timeIndex = timeIndex;

  return 0;
}

int ANCFSystem::setupPreconditioner() {
  cudaSetDevice(deviceIndex);

  ANCFSystem::updateInternalForces();
  delete mySolver;
  mySolver = new SpikeSolver(partitions, solverOptions);
  mySolver->setup(lhs);
  precUpdated = true;
  //printf("%f (Device %d): Update preconditioner\n", time, deviceIndex);

  return 0;
}

int ANCFSystem::updatePreconditioner() {
  cudaSetDevice(deviceIndex);

  ANCFSystem::updateInternalForces();
  mySolver->update(lhs.values);
  precUpdated = true;

  return 0;
}

int ANCFSystem::DoTimeStep() {
  cudaSetDevice(deviceIndex);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  stepKrylovIterations = 0;
  precUpdated = false;

  // update q and q_dot for initial guess
  cusp::blas::axpbypcz(pnew, vnew, anew, pnew, 1, h, .5 * h * h);
  cusp::blas::axpby(vnew, anew, vnew, 1, h);

  // Force a preconditioner update if needed
  if ((preconditionerUpdateModulus > 0) && (timeIndex % preconditionerUpdateModulus == 0)) {
    setupPreconditioner();
    printf("Preconditioner updated (step condition)!\n");
  }

  // Perform Newton iterations
  int it;
  for (it = 0; it < maxNewtonIterations; it++) {
    ANCFSystem::updatePhi();
    cusp::multiply(phiq, lambda, phiqlam);
    ANCFSystem::resetLeftHandSideMatrix();
    cusp::multiply(lhs_mass, anew, eTop); //cusp::multiply(mass,anew,eTop);
    ANCFSystem::updateInternalForces();
    cusp::blas::axpbypcz(eTop, fapp, fint, eTop, 1, -1, 1);
    cusp::blas::axpby(eTop, fext, eTop, 1, -1);
    cusp::blas::axpy(phiqlam, eTop, 1);
    cusp::blas::copy(phi, eBottom);

    // SOLVE THE LINEAR SYSTEM USING SPIKE
    cusp::blas::fill(delta, 0); // very important
    //stencil lhsStencil(anewAll.size(), lhs_mass, lhs_phiq, lhsVec);

    bool success = mySolver->solve(*m_spmv, eAll, delta);
    spike::Stats stats = mySolver->getStats();

    if(!success) {
      printf("**********  DUMP DATA **************\n");

      char filename[100];

      sprintf(filename, "./data/lhs%d.mtx", timeIndex);
      cusp::io::write_matrix_market_file(lhs, filename);

      sprintf(filename, "./data/rhs%d.mtx", timeIndex);
      cusp::io::write_matrix_market_file(eAll, filename);

      sprintf(filename, "./data/stats%d.txt", timeIndex);
      ofstream file(filename);
      file << "Code: " << mySolver->getMonitorCode();
      file << "  " << mySolver->getMonitorMessage() << std::endl;
      file << "Number of iterations = " << stats.numIterations << std::endl;
      file << "RHS norm             = " << stats.rhsNorm << std::endl;
      file << "Residual norm        = " << stats.residualNorm << std::endl;
      file << "Rel. residual norm   = " << stats.relResidualNorm << std::endl;
      file.close();

      int code = mySolver->getMonitorCode();
      if (code == -1 || code == -2) {

        //// TODO:  clean this up...

        std::cout << "STOP" << std::endl;
        exit(0);
      }
    }

    spikeSolveTime[it] = stats.timeSolve;
    spikeNumIter[it] = stats.numIterations;
    stepKrylovIterations += stats.numIterations;
    // END SOLVE THE LINEAR SYSTEM

    // update anew
    cusp::blas::axpy(delta, anewAll, -1);

    // update vnew
    cusp::blas::axpbypcz(v, a, anew, vnew, 1, h * (1 - gammaHHT), h * gammaHHT);

    // update pnew
    cusp::blas::axpbypcz(v, a, anew, pnew, h, h * h * .5 * (1 - 2 * betaHHT), h * h * .5 * 2 * betaHHT);
    cusp::blas::axpy(p, pnew, 1);

    // Calculate infinity norm of the correction and check for convergence
    double delta_nrm = cusp::blas::nrmmax(delta);

    //printf("         Krylov solver: %8.2f ms    %.2f iterations     ||delta||_inf = %e\n",
    //  stats.timeSolve, stats.numIterations, delta_nrm);

    if (delta_nrm <= tol)
      break;
  }

  // Number of Newton iterations and average number of Krylov iterations
  stepNewtonIterations = it + 1;
  float avgKrylov = stepKrylovIterations / stepNewtonIterations;

  if ((preconditionerMaxKrylovIterations > 0) && (avgKrylov > preconditionerMaxKrylovIterations)) {
    setupPreconditioner();
    printf("Preconditioner updated! (krylov condition)\n");
  }

  cusp::copy(anew, a);
  cusp::copy(vnew, v);
  cusp::copy(pnew, p);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  stepTime = elapsedTime;
  timeToSimulate += elapsedTime / 1000.0;

  p_h = p_d;

//  char filename[100];
//  sprintf(filename, "./data/lhs%d.txt",timeIndex);
//  cusp::io::write_matrix_market_file(lhs, filename);
//
//  char filename1[100];
//  sprintf(filename1, "./data/rhs%d.txt",timeIndex);
//  cusp::io::write_matrix_market_file(eAll, filename1);

  //printf("%f (Device %d), Elapsed time = %8.2f ms, Newton = %2d, Ave. Krylov Per Newton = %.2f\n",
  //       time, deviceIndex, elapsedTime, stepNewtonIterations, avgKrylov);

  time += h;
  timeIndex++;

  return 0;
}

float3 ANCFSystem::getXYZPosition(int elementIndex, double xi) {
  cudaSetDevice(deviceIndex);

  double a = elements[elementIndex].getLength_l();
  double* p = CASTD1(p_h);
  p = &p[12 * elementIndex];
  float3 pos;

  pos.x = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[0]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[3]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[6]
      + a * (-xi * xi + pow(xi, 3)) * p[9];
  pos.y = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[1]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[4]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[7]
      + a * (-xi * xi + pow(xi, 3)) * p[10];
  pos.z = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[2]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[5]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[8]
      + a * (-xi * xi + pow(xi, 3)) * p[11];

  return pos;
}

float3 ANCFSystem::getXYZVelocity(int elementIndex, double xi) {
  cudaSetDevice(deviceIndex);

  double a = elements[elementIndex].getLength_l();
  double* p = CASTD1(v_h);
  p = &p[12 * elementIndex];
  float3 pos;

  pos.x = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[0]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[3]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[6]
      + a * (-xi * xi + pow(xi, 3)) * p[9];
  pos.y = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[1]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[4]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[7]
      + a * (-xi * xi + pow(xi, 3)) * p[10];
  pos.z = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[2]
      + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[5]
      + (3 * xi * xi - 2 * pow(xi, 3)) * p[8]
      + a * (-xi * xi + pow(xi, 3)) * p[11];

  return pos;
}

float3 ANCFSystem::getXYZPositionParticle(int index) {
  cudaSetDevice(deviceIndex);

  return make_float3(pParticle_h[3 * index], pParticle_h[3 * index + 1],
      pParticle_h[3 * index + 2]);
}

float3 ANCFSystem::getXYZVelocityParticle(int index) {
  cudaSetDevice(deviceIndex);

  return make_float3(vParticle_h[3 * index], vParticle_h[3 * index + 1],
      vParticle_h[3 * index + 2]);
}

int ANCFSystem::saveLHS() {
  cudaSetDevice(deviceIndex);

  posFile.open("../lhs.dat");
  posFile << "symmetric" << endl;
  posFile << anew_h.size() << " " << anew_h.size() << " " << lhsI_h.size()
      << endl;
  for (int i = 0; i < lhsI_h.size(); i++) {
    posFile << lhsI_h[i] << " " << lhsJ_h[i] << " " << lhs_h[i] << endl;
  }
  posFile.close();

  return 0;
}

int ANCFSystem::writeToFile(string fileName) {
  cudaSetDevice(deviceIndex);

  posFile.open(fileName.c_str());
  p_h = p_d;
  double* posAll = CASTD1(p_h);
  double* pos;
  double l;
  double r;
  posFile << elements.size() << ", " << deviceIndex << ", " << endl;
  for (int i = 0; i < elements.size(); i++) {
    l = elements[i].getLength_l();
    r = elements[i].getRadius();
    pos = &posAll[12 * i];
    posFile << r << ", " << l;
    for (int i = 0; i < 12; i++)
      posFile << ", " << pos[i];
    posFile << ", " << endl;
  }
  posFile.close();

  return 0;
}
