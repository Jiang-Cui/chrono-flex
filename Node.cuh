/*
 * Node.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef NODE_CUH_
#define NODE_CUH_

#include "include.cuh"

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

#endif /* NODE_CUH_ */
