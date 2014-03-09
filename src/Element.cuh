/*
 * Element.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef ELEMENT_CUH_
#define ELEMENT_CUH_

#include "include.cuh"
#include "Node.cuh"

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
		this->node1 = Node(1, 0, 0, 1, 0, 0);
		this->r = .02;
		this->nu = .3;
		this->E = 2.e7;
		this->rho = 1150.0;
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
	Element(Node node0, Node node1, double r, double nu, double E, double rho) {
		this->node0 = node0;
		this->node1 = node1;
		this->r = r;
		this->l = getLength(node0, node1);
		this->nu = nu;
		this->E = E;
		this->rho = rho;
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

#endif /* ELEMENT_CUH_ */
