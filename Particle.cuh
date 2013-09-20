/*
 * Particle.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PARTICLE_CUH_
#define PARTICLE_CUH_

#include "include.cuh"

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

#endif /* PARTICLE_CUH_ */
