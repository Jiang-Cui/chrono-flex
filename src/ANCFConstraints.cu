#include "include.cuh"
#include "ANCFSystem.cuh"

int ANCFSystem::addConstraint_AbsoluteX(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteY(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEY);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteZ(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEZ);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDX1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDX1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDY1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDY1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDZ1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDZ1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeX(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeY(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeZ(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeDX1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDX1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeDY1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDY1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_RelativeDZ1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDZ1);
	ANCFSystem::constraints.push_back(constraint);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteFixed(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	ANCFSystem::constraints.push_back(constraint);
	Constraint constraint2(nodeNum,CONSTRAINTABSOLUTEY);
	ANCFSystem::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum,CONSTRAINTABSOLUTEZ);
	ANCFSystem::constraints.push_back(constraint3);

	Constraint constraint4(nodeNum,CONSTRAINTABSOLUTEDX1);
	ANCFSystem::constraints.push_back(constraint4);
	Constraint constraint5(nodeNum,CONSTRAINTABSOLUTEDY1);
	ANCFSystem::constraints.push_back(constraint5);
	Constraint constraint6(nodeNum,CONSTRAINTABSOLUTEDZ1);
	ANCFSystem::constraints.push_back(constraint6);

	return 0;
}

int ANCFSystem::addConstraint_RelativeFixed(int nodeNum1,int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	ANCFSystem::constraints.push_back(constraint);
	Constraint constraint2(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	ANCFSystem::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	ANCFSystem::constraints.push_back(constraint3);

	Constraint constraint4(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDX1);
	ANCFSystem::constraints.push_back(constraint4);
	Constraint constraint5(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDY1);
	ANCFSystem::constraints.push_back(constraint5);
	Constraint constraint6(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDZ1);
	ANCFSystem::constraints.push_back(constraint6);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteSpherical(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	ANCFSystem::constraints.push_back(constraint);
	Constraint constraint2(nodeNum,CONSTRAINTABSOLUTEY);
	ANCFSystem::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum,CONSTRAINTABSOLUTEZ);
	ANCFSystem::constraints.push_back(constraint3);

	return 0;
}

int ANCFSystem::addConstraint_RelativeSpherical(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	ANCFSystem::constraints.push_back(constraint);
	Constraint constraint2(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	ANCFSystem::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	ANCFSystem::constraints.push_back(constraint3);

	return 0;
}

int ANCFSystem::addConstraint_AbsoluteX(Element& element, int node_local)
{
	addConstraint_AbsoluteX(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteY(Element& element, int node_local)
{
	addConstraint_AbsoluteY(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteZ(Element& element, int node_local)
{
	addConstraint_AbsoluteZ(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDX1(Element& element, int node_local)
{
	addConstraint_AbsoluteDX1(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDY1(Element& element, int node_local)
{
	addConstraint_AbsoluteDY1(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteDZ1(Element& element, int node_local)
{
	addConstraint_AbsoluteDZ1(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteFixed(Element& element, int node_local)
{
	addConstraint_AbsoluteFixed(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_AbsoluteSpherical(Element& element, int node_local)
{
	addConstraint_AbsoluteSpherical(element.getElementIndex()*2+node_local);
	return 0;
}

int ANCFSystem::addConstraint_RelativeX(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeX(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeY(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeY(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeZ(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeZ(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeDX1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDX1(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeDY1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDY1(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeDZ1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDZ1(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeFixed(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeFixed(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}

int ANCFSystem::addConstraint_RelativeSpherical(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeSpherical(element1.getElementIndex()*2+node_local1,element2.getElementIndex()*2+node_local2);
	return 0;
}
