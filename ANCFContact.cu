#include "include.cuh"

int ANCFSystem::updateBoundingBoxes_CPU()
{
	int offset = 0;
	// push minimum aabb
	for(int i=0;i<particles.size();i++)
	{
		aabb_data_h[i] = make_float3(pParticle_h[3*i],pParticle_h[3*i+1],pParticle_h[3*i+2])-1.1*particles[i].getRadius()*make_float3(1,1,1);
	}
	for(int i=0;i<elements.size();i++)
	{
		offset = 12*i;
		aabb_data_h[i+particles.size()] = make_float3(min(p_h[offset],p_h[offset+6]),min(p_h[offset+1],p_h[offset+7]),min(p_h[offset+2],p_h[offset+8]))-3*elements[i].getRadius()*make_float3(1,1,1);
	}

	// push maximum aabb
	for(int i=0;i<particles.size();i++)
	{
		aabb_data_h[i+particles.size()+elements.size()] = make_float3(pParticle_h[3*i],pParticle_h[3*i+1],pParticle_h[3*i+2])+1.1*particles[i].getRadius()*make_float3(1,1,1);
	}
	for(int i=0;i<elements.size();i++)
	{
		offset = 12*i;
		aabb_data_h[i+particles.size()+elements.size()+particles.size()] = make_float3(max(p_h[offset],p_h[offset+6]),max(p_h[offset+1],p_h[offset+7]),max(p_h[offset+2],p_h[offset+8]))+3*elements[i].getRadius()*make_float3(1,1,1);
	}
	aabb_data_d = aabb_data_h;
	return 0;
}
/*
__global__ void updateBoundingBox(float3* aabb_data, double* p, Material* materials, int numElements)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numElements)
	{
		double r = materials[i].r;

		p = &p[12*i];

		aabb_data[i] = make_float3(min(p[0],p[6]),min(p[1],p[7]),min(p[2],p[8]))-3*r*make_float3(1,1,1);
		aabb_data[i+numElements] = make_float3(max(p[0],p[6]),max(p[1],p[7]),max(p[2],p[8]))+3*r*make_float3(1,1,1);
	}
}

int ANCFSystem::updateBoundingBoxes()
{
	updateBoundingBox<<<dimGridElement,dimBlockElement>>>(CASTF3(aabb_data_d),CASTD1(p_d),CASTM1(materials_d),elements.size());
	return 0;
}
*/
int ANCFSystem::initializeBoundingBoxes_CPU()
{
	int offset = 0;
	// push minimum aabb
	for(int i=0;i<particles.size();i++)
	{
		aabb_data_h.push_back(make_float3(pParticle_h[3*i],pParticle_h[3*i+1],pParticle_h[3*i+2])-1.1*particles[i].getRadius()*make_float3(1,1,1));
	}
	for(int i=0;i<elements.size();i++)
	{
		offset = 12*i;
		if(p_h[offset]>p_h[offset+6]) offset+=6;
		aabb_data_h.push_back(make_float3(p_h[offset],p_h[offset+1],p_h[offset+2])-2*elements[i].getRadius()*make_float3(1,1,1));
	}

	// push maximum aabb
	for(int i=0;i<particles.size();i++)
	{
		aabb_data_h.push_back(make_float3(pParticle_h[3*i],pParticle_h[3*i+1],pParticle_h[3*i+2])+1.1*particles[i].getRadius()*make_float3(1,1,1));
	}
	for(int i=0;i<elements.size();i++)
	{
		offset = 12*i;
		if(p_h[offset]<p_h[offset+6]) offset+=6;
		aabb_data_h.push_back(make_float3(p_h[offset],p_h[offset+1],p_h[offset+2])+2*elements[i].getRadius()*make_float3(1,1,1));
	}
	aabb_data_d = aabb_data_h;
	return 0;
}

int ANCFSystem::applyContactForce_CPU(int beamIndex, int particleIndex, double penetration, double xi, float3 normal)
{
	double l1 = elements[beamIndex].getLength_l();
	double R1 = elements[beamIndex].getRadius();
	double nu1 = elements[beamIndex].getNu();
	double E1 = elements[beamIndex].getElasticModulus();
	double sigma1 = (1-nu1*nu1)/E1;
	//cout << l1 << " "<< R1 << " " << nu1 << " " << E1 << " " << sigma1 << endl;

	double R2 = particles[particleIndex].getRadius();
	double nu2 = particles[particleIndex].getNu();
	double E2 = particles[particleIndex].getElasticModulus();
	double sigma2 = (1-nu2*nu2)/E2;
	//cout << R2 << " " << nu2 << " " << E2 << " " << sigma2 << endl;

	double K = 4.0/(3.0*(sigma1+sigma2))*sqrt(R1*R2/(R1+R2));
	double n = .3;
	//cout << K << " " << penetration << endl;

	float3 v = getXYZVelocity(beamIndex,xi)-getXYZVelocityParticle(particleIndex);
	//cout << "v:" << v.x << " " << v.y << " " << v.z << endl;
	double b = 2500;

	float3 damping;
	damping.x = -b * normal.x * normal.x * v.x - b * normal.x * normal.y * v.y - b * normal.x * normal.z * v.z;
	damping.y = -b * normal.x * normal.y * v.x - b * normal.y * normal.y * v.y - b * normal.y * normal.z * v.z;
	damping.z = -b * normal.x * normal.z * v.x - b * normal.y * normal.z * v.y - b * normal.z * normal.z * v.z;
	//cout << "damping:" << damping.x << " " << damping.y << " " << damping.z << endl;

	float3 vct;
	vct.x = v.x - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.x;
	vct.y = v.y - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.y;
	vct.z = v.z - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.z;
	vct = normalize(vct);
	if(vct.x!=vct.x) vct = make_float3(0,0,0);
	//cout << "vct:" << vct.x << " " << vct.y << " " << vct.z << endl;

	double mu = .05;
	float3 force = K*pow(penetration,n)*(normal+mu*vct)-damping;
	//cout << "Force: " << force.x << " " << force.y << " " << force.z << endl;
	applyForce_CPU(beamIndex,l1,xi,-1*force);
	applyForceParticle_CPU(particleIndex,force);
	//cin.get();

	return 0;
}

int ANCFSystem::applyContactForceParticles_CPU(int particleIndex1, int particleIndex2, double penetration, float3 normal)
{
	double R1 = particles[particleIndex1].getRadius();
	double nu1 = particles[particleIndex1].getNu();
	double E1 = particles[particleIndex1].getElasticModulus();
	double sigma1 = (1-nu1*nu1)/E1;
	//cout << l1 << " "<< R1 << " " << nu1 << " " << E1 << " " << sigma1 << endl;

	double R2 = particles[particleIndex2].getRadius();
	double nu2 = particles[particleIndex2].getNu();
	double E2 = particles[particleIndex2].getElasticModulus();
	double sigma2 = (1-nu2*nu2)/E2;
	//cout << R2 << " " << nu2 << " " << E2 << " " << sigma2 << endl;

	double K = 4.0/(3.0*(sigma1+sigma2))*sqrt(R1*R2/(R1+R2));
	double n = .5;
	//cout << K << " " << penetration << endl;

	float3 v = getXYZVelocityParticle(particleIndex1)-getXYZVelocityParticle(particleIndex2);
	//cout << "v:" << v.x << " " << v.y << " " << v.z << endl;
	double b = 0;//2500;

	float3 damping;
	damping.x = -b * normal.x * normal.x * v.x - b * normal.x * normal.y * v.y - b * normal.x * normal.z * v.z;
	damping.y = -b * normal.x * normal.y * v.x - b * normal.y * normal.y * v.y - b * normal.y * normal.z * v.z;
	damping.z = -b * normal.x * normal.z * v.x - b * normal.y * normal.z * v.y - b * normal.z * normal.z * v.z;
	//cout << "damping:" << damping.x << " " << damping.y << " " << damping.z << endl;

	float3 vct;
	vct.x = v.x - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.x;
	vct.y = v.y - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.y;
	vct.z = v.z - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.z;
	vct = normalize(vct);
	if(vct.x!=vct.x) vct = make_float3(0,0,0);
	//cout << "vct:" << vct.x << " " << vct.y << " " << vct.z << endl;

	double mu = 0;//.05;
	float3 force = K*pow(penetration,n)*(normal+mu*vct)-damping;
	//cout << "Force: " << force.x << " " << force.y << " " << force.z << endl;
	//applyForce_CPU(beamIndex,l1,xi,-1*force);
	applyForceParticle_CPU(particleIndex1,-1*force);
	applyForceParticle_CPU(particleIndex2,force);
	//cin.get();

	return 0;
}

double detectCollisionsBetweenSpheres(double r1, float3 pos1, double r2, float3 pos2)
{
	double penetration = length(pos1-pos2)-r1-r2;
	if(penetration>0) penetration = 0;
	return -1*penetration;
}

__device__ float3 getXYZBeamGPU(int elementIndex, double xi, double* p, double a)
{
	float3 pos;

	pos.x = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[0] + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[3] + (3 * xi * xi - 2 * pow(xi, 3)) * p[6] + a * (-xi * xi + pow(xi, 3)) * p[9];
	pos.y = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[1] + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[4] + (3 * xi * xi - 2 * pow(xi, 3)) * p[7] + a * (-xi * xi + pow(xi, 3)) * p[10];
	pos.z = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[2] + a * (xi - 2 * xi * xi + pow(xi, 3)) * p[5] + (3 * xi * xi - 2 * pow(xi, 3)) * p[8] + a * (-xi * xi + pow(xi, 3)) * p[11];

	return pos;
}

__device__ float3 getXYZParticleGPU(int index, double* p)
{
	return make_float3(p[0],p[1],p[2]);
}

__device__ double detectCollisionsBetweenSpheresGPU(double r1, float3 pos1, double r2, float3 pos2)
{
	double penetration = length(pos1-pos2)-r1-r2;
	if(penetration>0) penetration = 0;
	return -1*penetration;
}

__global__ void countActualCollisions_GPU(double* p, double* pParticle, long long* potentialCollisions, uint* collisionCount, Material* materials, MaterialParticle* pMaterials, int numParticles, int numPotentialCollisions)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numPotentialCollisions)
	{
		uint count = 0;
		int2 collisionPair;
		collisionPair.x = int(potentialCollisions[i] >> 32);
		collisionPair.y = int(potentialCollisions[i] & 0xffffffff);

		if(collisionPair.x<numParticles&&collisionPair.y>=numParticles)
		{
			int beamIndex = collisionPair.y-numParticles;
			int particleIndex = collisionPair.x;

			p = &p[12*beamIndex];
			pParticle = &pParticle[3*particleIndex];

			double xiInc = 1/(static_cast<double>(materials[beamIndex].numContactPoints-1));
			for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
			{
				float3 beamPos = getXYZBeamGPU(beamIndex,xiInc*i,p,materials[beamIndex].l);
				float3 particlePos = getXYZParticleGPU(particleIndex,pParticle);
				double penetration = detectCollisionsBetweenSpheresGPU(materials[beamIndex].r, beamPos, pMaterials[particleIndex].r, particlePos);
				if (penetration) {
					count++;
				}
			}
		}
		if(collisionPair.y<numParticles&&collisionPair.x>=numParticles)
		{
			int beamIndex = collisionPair.x-numParticles;
			int particleIndex = collisionPair.y;

			p = &p[12*beamIndex];
			pParticle = &pParticle[3*particleIndex];


			double xiInc = 1/(static_cast<double>(materials[beamIndex].numContactPoints-1));
			for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
			{
				float3 beamPos = getXYZBeamGPU(beamIndex,xiInc*i,p,materials[beamIndex].l);
				float3 particlePos = getXYZParticleGPU(particleIndex,pParticle);
				double penetration = detectCollisionsBetweenSpheresGPU(materials[beamIndex].r, beamPos, pMaterials[particleIndex].r, particlePos);
				if (penetration) {
					count++;
				}
			}

		}
		if(collisionPair.y<numParticles&&collisionPair.x<numParticles)
		{
			int particleIndex1 = collisionPair.x;
			int particleIndex2 = collisionPair.y;

			float3 particlePos1 = getXYZParticleGPU(particleIndex1,&pParticle[3*particleIndex1]);
			float3 particlePos2 = getXYZParticleGPU(particleIndex2,&pParticle[3*particleIndex2]);
			double penetration = detectCollisionsBetweenSpheresGPU(pMaterials[particleIndex1].r, particlePos1, pMaterials[particleIndex2].r, particlePos2);
			if (penetration) {
				count++;
			}
		}
		collisionCount[i] = count;
	}
}

int ANCFSystem::countActualCollisions()
{
	countActualCollisions_GPU<<<dimGridCollision,dimBlockCollision>>>(CASTD1(p_d), CASTD1(pParticle_d), CASTLL(detector.potentialCollisions), CASTU1(collisionCounts_d), CASTM1(materials_d), CASTMP(pMaterials_d), particles.size(), detector.number_of_contacts_possible);
	return 0;
}

__global__ void populateCollisions_GPU(double* collisionAlongBeam, uint* collisionIndices1, uint* collisionIndices2, float3* normals, double* penetrations, double* p, double* pParticle, long long* potentialCollisions, uint* collisionCount, Material* materials, MaterialParticle* pMaterials, int numParticles, int numPotentialCollisions)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numPotentialCollisions)
	{
		uint startIndex = 0;
		if(i)
		{
			startIndex = collisionCount[i-1];
		}

		uint count = 0;
		int2 collisionPair;
		collisionPair.x = int(potentialCollisions[i] >> 32);
		collisionPair.y = int(potentialCollisions[i] & 0xffffffff);

		if(collisionPair.x<numParticles&&collisionPair.y>=numParticles)
		{
			int beamIndex = collisionPair.y-numParticles;
			int particleIndex = collisionPair.x;

			p = &p[12*beamIndex];
			pParticle = &pParticle[3*particleIndex];

			double xiInc = 1/(static_cast<double>(materials[beamIndex].numContactPoints-1));
			for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
			{
				float3 beamPos = getXYZBeamGPU(beamIndex,xiInc*i,p,materials[beamIndex].l);
				float3 particlePos = getXYZParticleGPU(particleIndex,pParticle);
				double penetration = detectCollisionsBetweenSpheresGPU(materials[beamIndex].r, beamPos, pMaterials[particleIndex].r, particlePos);
				if (penetration) {
					normals[startIndex] = normalize(particlePos-beamPos);
					penetrations[startIndex] = penetration;
					collisionIndices1[startIndex] = collisionPair.x;
					collisionIndices2[startIndex] = collisionPair.y;
					collisionAlongBeam[startIndex] = xiInc*i;
					startIndex++;
					count++;
				}
			}
		}
		if(collisionPair.y<numParticles&&collisionPair.x>=numParticles)
		{
			int beamIndex = collisionPair.x-numParticles;
			int particleIndex = collisionPair.y;

			p = &p[12*beamIndex];
			pParticle = &pParticle[3*particleIndex];


			double xiInc = 1/(static_cast<double>(materials[beamIndex].numContactPoints-1));
			for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
			{
				float3 beamPos = getXYZBeamGPU(beamIndex,xiInc*i,p,materials[beamIndex].l);
				float3 particlePos = getXYZParticleGPU(particleIndex,pParticle);
				double penetration = detectCollisionsBetweenSpheresGPU(materials[beamIndex].r, beamPos, pMaterials[particleIndex].r, particlePos);
				if (penetration) {
					normals[startIndex] = normalize(particlePos-beamPos);
					penetrations[startIndex] = penetration;
					collisionIndices1[startIndex] = collisionPair.x;
					collisionIndices2[startIndex] = collisionPair.y;
					collisionAlongBeam[startIndex] = xiInc*i;
					startIndex++;
					count++;
				}
			}

		}
		if(collisionPair.y<numParticles&&collisionPair.x<numParticles)
		{
			int particleIndex1 = collisionPair.x;
			int particleIndex2 = collisionPair.y;

			float3 particlePos1 = getXYZParticleGPU(particleIndex1,&pParticle[3*particleIndex1]);
			float3 particlePos2 = getXYZParticleGPU(particleIndex2,&pParticle[3*particleIndex2]);
			double penetration = detectCollisionsBetweenSpheresGPU(pMaterials[particleIndex1].r, particlePos1, pMaterials[particleIndex2].r, particlePos2);
			if (penetration) {
				normals[startIndex] = normalize(particlePos2-particlePos1);
				penetrations[startIndex] = penetration;
				collisionIndices1[startIndex] = collisionPair.x;
				collisionIndices2[startIndex] = collisionPair.y;
				collisionAlongBeam[startIndex] = 0;
				startIndex++;
				count++;
			}
		}
	}
}

int ANCFSystem::populateCollisions()
{
	populateCollisions_GPU<<<dimGridCollision,dimBlockCollision>>>(CASTD1(collisionAlongBeam_d), CASTU1(collisionIndices1_d), CASTU1(collisionIndices2_d), CASTF3(collisionNormals_d), CASTD1(collisionPenetrations_d), CASTD1(p_d), CASTD1(pParticle_d), CASTLL(detector.potentialCollisions), CASTU1(collisionCounts_d), CASTM1(materials_d), CASTMP(pMaterials_d), particles.size(), detector.number_of_contacts_possible);
	return 0;
}

__global__ void accumulateContactForces_GPU(double* fParticle, double* collisionAlongBeam, uint* collisionIndices1, uint* collisionIndices2, float3* normals, double* penetrations, double* p, double* pParticle, double* vBeam, double* vParticle, uint* collisionCount, Material* materials, MaterialParticle* pMaterials, int numParticles, int numBodiesInContact)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<numBodiesInContact)
	{
		uint startIndex = 0;
		if(i)
		{
			startIndex = collisionCount[i-1];
		}
		uint endIndex = collisionCount[i];

		float3 particleForce = make_float3(0,0,0);
		uint bodyIndex = collisionIndices1[i];

		for(int j=startIndex;j<endIndex;j++)
		{
			int2 collisionPair = make_int2(collisionIndices1[j],collisionIndices2[j]);
			if((collisionPair.x<numParticles&&collisionPair.y>=numParticles)||(collisionPair.y<numParticles&&collisionPair.x>=numParticles))
			{
				int beamIndex = max(collisionPair.x,collisionPair.y)-numParticles;
				int particleIndex = min(collisionPair.x,collisionPair.y);

				double l1 = materials[beamIndex].l;
				double R1 = materials[beamIndex].r;
				double nu1 = materials[beamIndex].nu;
				double E1 = materials[beamIndex].E;
				double sigma1 = (1-nu1*nu1)/E1;
				//cout << l1 << " "<< R1 << " " << nu1 << " " << E1 << " " << sigma1 << endl;

				double R2 = pMaterials[particleIndex].r;
				double nu2 = pMaterials[particleIndex].nu;
				double E2 = pMaterials[particleIndex].E;
				double sigma2 = (1-nu2*nu2)/E2;
				//cout << R2 << " " << nu2 << " " << E2 << " " << sigma2 << endl;

				double K = 4.0/(3.0*(sigma1+sigma2))*sqrt(R1*R2/(R1+R2));
				double n = .3;
				//cout << K << " " << penetration << endl;

				float3 v = getXYZBeamGPU(beamIndex,collisionAlongBeam[j],&vBeam[12*beamIndex],materials[beamIndex].l)-getXYZParticleGPU(particleIndex,&pParticle[3*particleIndex]);
				//cout << "v:" << v.x << " " << v.y << " " << v.z << endl;
				double b = 2500;

				float3 damping;
				float3 normal = normals[j];
				damping.x = -b * normal.x * normal.x * v.x - b * normal.x * normal.y * v.y - b * normal.x * normal.z * v.z;
				damping.y = -b * normal.x * normal.y * v.x - b * normal.y * normal.y * v.y - b * normal.y * normal.z * v.z;
				damping.z = -b * normal.x * normal.z * v.x - b * normal.y * normal.z * v.y - b * normal.z * normal.z * v.z;
				//cout << "damping:" << damping.x << " " << damping.y << " " << damping.z << endl;

				float3 vct;
				vct.x = v.x - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.x;
				vct.y = v.y - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.y;
				vct.z = v.z - (normal.x * v.x + normal.y * v.y + normal.z * v.z) * normal.z;
				vct = normalize(vct);
				if(vct.x!=vct.x) vct = make_float3(0,0,0);
				//cout << "vct:" << vct.x << " " << vct.y << " " << vct.z << endl;

				double mu = .05;
				float3 force = K*pow(penetrations[j],n)*(normal+mu*vct)-damping;
				particleForce+=force;
			}
			else//(collisionPair.y<numParticles&&collisionPair.x<numParticles)
			{
				int particleIndex1 = collisionPair.x;
				int particleIndex2 = collisionPair.y;
			}
		}
		fParticle[3*bodyIndex+0]+=particleForce.x;
		fParticle[3*bodyIndex+1]+=particleForce.y;
		fParticle[3*bodyIndex+2]+=particleForce.z;

	}
}

int ANCFSystem::accumulateContactForces(int numBodiesInContact)
{
	dimBlockCollision.x = BLOCKDIMCOLLISION;
	dimGridCollision.x = (int)ceil(((double)(numBodiesInContact))/((double)BLOCKDIMCOLLISION));
	accumulateContactForces_GPU<<<dimGridCollision,dimBlockCollision>>>(CASTD1(fParticle_d), CASTD1(collisionAlongBeam_d), CASTU1(collisionIndices1_d), CASTU1(collisionIndices2_d), CASTF3(collisionNormals_d), CASTD1(collisionPenetrations_d), CASTD1(p_d), CASTD1(pParticle_d), CASTD1(v_d), CASTD1(vParticle_d), CASTU1(collisionCounts_d), CASTM1(materials_d), CASTMP(pMaterials_d), particles.size(), numBodiesInContact);
	return 0;
}

int ANCFSystem::performNarrowphaseCollisionDetection()
{

	// Step 0: Resize the collisionCounts vector to equal the potentialCollisions vector(THRUST)
	collisionCounts_d.resize(detector.number_of_contacts_possible);
	dimBlockCollision.x = BLOCKDIMCOLLISION;
	dimGridCollision.x = (int)ceil(((double)(detector.number_of_contacts_possible))/((double)BLOCKDIMCOLLISION));

	// Step 1: Count the actual collisions, store in corresponding indices to potentialCollisions (CUSTOM)
	countActualCollisions();

	// Step 2: Perform a prefix sum to determine start and end indices of the arrays, store the total in numActualCollisions (THRUST)
	Thrust_Inclusive_Scan_Sum(collisionCounts_d,numActualCollisions);
//	cout << numActualCollisions << endl;
//	collisionCounts_h = collisionCounts_d;
//	if(numActualCollisions) {
//	for(int i=0;i<collisionCounts_h.size();i++) cout << i << " " << collisionCounts_h[i] << endl;
//	cin.get();
//	}

	// Step 3: Resize collision normal and collision penetration vectors (THRUST)
	collisionIndices1_d.resize(numActualCollisions);
	collisionIndices2_d.resize(numActualCollisions);
	collisionNormals_d.resize(numActualCollisions);
	collisionPenetrations_d.resize(numActualCollisions);
	collisionAlongBeam_d.resize(numActualCollisions);

	// Step 4: Populate the collision normal and collision penetration vectors with actual collisions based on start and end indices(CUSTOM)
	populateCollisions();
//	collisionIndices1_h = collisionIndices1_d;
//	collisionIndices2_h = collisionIndices2_d;
//	collisionPenetrations_h = collisionPenetrations_d;
//	collisionNormals_h = collisionNormals_d;
//	if(numActualCollisions) cout << "Collisions:" << endl;
//
//		for (int i = 0; i < numActualCollisions; i++)
//		{
//			cout << "Collisions " << i << ": (" << collisionIndices1_h[i] << ", " << collisionIndices2_h[i] << ")  Normal = (" << collisionNormals_h[i].x << ", " << collisionNormals_h[i].y << ", " << collisionNormals_h[i].z << ") Penetration =  " << collisionPenetrations_h[i] << endl;
//		}
//		cin.get();

	// Step 5: Sort the collision normals and collision penetration vectors by the first collision body index (THRUST)
//	thrust::sort_by_key(collisionIndices1_d.begin(), collisionIndices1_d.end(),
//			thrust::make_zip_iterator(
//					thrust::make_tuple(collisionIndices2_d.begin(), collisionNormals_d.begin(),collisionPenetrations_d.begin()
//							)));
//		collisionIndices1_h = collisionIndices1_d;
//		collisionIndices2_h = collisionIndices2_d;
//		collisionPenetrations_h = collisionPenetrations_d;
//		collisionNormals_h = collisionNormals_d;
//		if(numActualCollisions) cout << "Collisions sorted by first index:" << endl;
//
//			for (int i = 0; i < numActualCollisions; i++)
//			{
//				cout << "Collisions " << i << ": (" << collisionIndices1_h[i] << ", " << collisionIndices2_h[i] << ")  Normal = (" << collisionNormals_h[i].x << ", " << collisionNormals_h[i].y << ", " << collisionNormals_h[i].z << ") Penetration =  " << collisionPenetrations_h[i] << endl;
//			}

	// Step 6: Reduce the collision body indices (and count them) (THRUST)
	//uint numBodiesInContact = 0;
	//Thrust_Reduce_By_KeyA(numBodiesInContact,collisionIndices1_d,collisionCounts_d);
//	for(int i=0;i<numBodiesInContact;i++)
//	{
//		cout << collisionCounts_d[i] << endl;
//	}

	// Step 7: Perform a prefix sum to determine the start and end indices of the collisions for each body (THRUST)
	//uint checkNumActualCollisions = 0;
	//Thrust_Inclusive_Scan_Sum(collisionCounts_d,checkNumActualCollisions);

	// Step 8: Accumulate the forces, add to corresponding force vectors (CUSTOM)
	//thrust::fill(fParticle_d.begin(),fParticle_d.end(),0);
	//accumulateContactForces(numBodiesInContact);

	// STEP 9: Sort the collision normals and collision penetration vectors by the second collision body index (THRUST)

	// Step 10: Reduce the collision body indices (and count them) (THRUST)

	// Step 11: Perform a prefix sum to determine the start and end indices of the collisions for each body (THRUST)

	// STEP 12: Accumulate the forces add to corresponding force vectors (CUSTOM)

	return 0;
}

int ANCFSystem::performNarrowphaseCollisionDetection_CPU(long long potentialCollisions)
{
	int2 collisionPair;
	collisionPair.x = int(potentialCollisions >> 32);
	collisionPair.y = int(potentialCollisions & 0xffffffff);
	//cout << "\tCollision Pair: " << collisionPair.x << " " << collisionPair.y << endl;
	int count = 0;

	if(collisionPair.x<particles.size()&&collisionPair.y>=particles.size())
	{
		int beamIndex = collisionPair.y-particles.size();
		int particleIndex = collisionPair.x;

		double xiInc = 1/(static_cast<double>(numContactPoints-1));
		for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
		{
			float3 beamPos = getXYZPosition(beamIndex,xiInc*i);
			float3 particlePos = getXYZPositionParticle(particleIndex);
			double penetration = detectCollisionsBetweenSpheres(elements[beamIndex].getRadius(), beamPos, particles[particleIndex].getRadius(), particlePos);
			if (penetration) {
				float3 normal = normalize(particlePos-beamPos);
				//cout << "Normal: " << normal.x << " " << normal.y << " " << normal.z << endl;
				applyContactForce_CPU(beamIndex, particleIndex, penetration, xiInc*i, normal);
				count++;
			}
		}
	}
	if(collisionPair.y<particles.size()&&collisionPair.x>=particles.size())
	{
		int beamIndex = collisionPair.x-particles.size();
		int particleIndex = collisionPair.y;

		double xiInc = 1/(static_cast<double>(numContactPoints-1));
		for(int i=1;i<materials[beamIndex].numContactPoints-1;i++)
		{
			float3 beamPos = getXYZPosition(beamIndex,xiInc*i);
			float3 particlePos = getXYZPositionParticle(particleIndex);
			double penetration = detectCollisionsBetweenSpheres(elements[beamIndex].getRadius(), beamPos, particles[particleIndex].getRadius(), particlePos);
			if (penetration) {
				applyContactForce_CPU(beamIndex, particleIndex, penetration, xiInc*i, normalize(particlePos-beamPos));
				count++;
			}
		}

	}
	if(collisionPair.y<particles.size()&&collisionPair.x<particles.size())
	{
		int particleIndex1 = collisionPair.x;
		int particleIndex2 = collisionPair.y;

		float3 particlePos1 = getXYZPositionParticle(particleIndex1);
		float3 particlePos2 = getXYZPositionParticle(particleIndex2);
		double penetration = detectCollisionsBetweenSpheres(particles[particleIndex1].getRadius(), particlePos1, particles[particleIndex2].getRadius(), particlePos2);
		if (penetration) {
			applyContactForceParticles_CPU(particleIndex1, particleIndex2, penetration, normalize(particlePos2-particlePos1));
			count++;
		}
	}

	return count;
}

int ANCFSystem::accumulateContactForces_CPU()
{
	thrust::fill(fcon_h.begin(),fcon_h.end(),0);
	thrust::fill(fParticle_h.begin(),fParticle_h.end(),0);
	collisionIndices1_h = collisionIndices1_d;
	collisionIndices2_h = collisionIndices2_d;
	collisionPenetrations_h = collisionPenetrations_d;
	collisionNormals_h = collisionNormals_d;
	collisionAlongBeam_h = collisionAlongBeam_d;

	for(int i = 0;i<numActualCollisions;i++)
	{
		int2 collisionPair = make_int2(collisionIndices1_h[i],collisionIndices2_h[i]);
		if((collisionPair.x<particles.size()&&collisionPair.y>=particles.size())||(collisionPair.y<particles.size()&&collisionPair.x>=particles.size()))
		{
			int beamIndex = max(collisionPair.x,collisionPair.y)-particles.size();
			int particleIndex = min(collisionPair.x,collisionPair.y);
			applyContactForce_CPU(beamIndex,particleIndex,collisionPenetrations_h[i],collisionAlongBeam_h[i],collisionNormals_h[i]);
		}
		else
		{
			applyContactForceParticles_CPU(collisionPair.x,collisionPair.y,collisionPenetrations_h[i],collisionNormals_h[i]);
		}
	}
	fcon_d = fcon_h;
	fParticle_d = fParticle_h;

}

int ANCFSystem::detectCollisions_CPU()
{
//	thrust::fill(fcon_h.begin(),fcon_h.end(),0);
//	thrust::fill(fParticle_h.begin(),fParticle_h.end(),0);
//	potentialCollisions_h = detector.potentialCollisions;
//	int count = 0;
//	for(int i=0;i<detector.number_of_contacts_possible;i++)
//	{
//		count += performNarrowphaseCollisionDetection_CPU(potentialCollisions_h[i]);
//	}
//	fcon_d = fcon_h;
//	fParticle_d = fParticle_h;

	//cout << count << endl;

	performNarrowphaseCollisionDetection();
	accumulateContactForces_CPU();

	return 0;
}

int ANCFSystem::applyForce_CPU(int elementIndex, double l, double xi, float3 force)
{
	fcon_h[0 + 12 * elementIndex]  += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.x;
	fcon_h[1 + 12 * elementIndex]  += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.y;
	fcon_h[2 + 12 * elementIndex]  += (1 - 3 * xi * xi + 2 * pow(xi, 3)) * force.z;
	fcon_h[3 + 12 * elementIndex]  += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.x;
	fcon_h[4 + 12 * elementIndex]  += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.y;
	fcon_h[5 + 12 * elementIndex]  += l * (xi - 2 * xi * xi + pow(xi, 3)) * force.z;
	fcon_h[6 + 12 * elementIndex]  += (3 * xi * xi - 2 * pow(xi, 3)) * force.x;
	fcon_h[7 + 12 * elementIndex]  += (3 * xi * xi - 2 * pow(xi, 3)) * force.y;
	fcon_h[8 + 12 * elementIndex]  += (3 * xi * xi - 2 * pow(xi, 3)) * force.z;
	fcon_h[9 + 12 * elementIndex]  += l * (-xi * xi + pow(xi, 3)) * force.x;
	fcon_h[10 + 12 * elementIndex] += l * (-xi * xi + pow(xi, 3)) * force.y;
	fcon_h[11 + 12 * elementIndex] += l * (-xi * xi + pow(xi, 3)) * force.z;

	return 0;
}

int ANCFSystem::applyForceParticle_CPU(int particleIndex, float3 force)
{
	fParticle_h[0 + 3 * particleIndex] += force.x;
	fParticle_h[1 + 3 * particleIndex] += force.y;
	fParticle_h[2 + 3 * particleIndex] += force.z;
	//cout << force.x << " " << force.y << " " << force.z << endl;
	return 0;
}

