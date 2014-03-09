/*
 * Constraint.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef CONSTRAINT_CUH_
#define CONSTRAINT_CUH_

#include "include.cuh"

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

#endif /* CONSTRAINT_CUH_ */
