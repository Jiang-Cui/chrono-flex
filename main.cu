#include "include.cuh"
#include "ANCFSystem.cuh"
#include "Element.cuh"
#include "Node.cuh"
#include "Particle.cuh"

bool updateDraw = 1;
bool showSphere = 1;

ANCFSystem sys;
OpenGLCamera oglcamera(camreal3(1,5,0),camreal3(0,0,0),camreal3(0,1,0),.01);

//RENDERING STUFF
void changeSize(int w, int h) {
	if(h == 0) {h = 1;}
	float ratio = 1.0* w / h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluPerspective(45,ratio,.1,1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,0.0,		0.0,0.0,-7,		0.0f,1.0f,0.0f);
}

void initScene(){
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
	glClearColor (1.0, 1.0, 1.0, 0.0);
	glShadeModel (GL_SMOOTH);
	glEnable(GL_COLOR_MATERIAL);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable (GL_POINT_SMOOTH);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint (GL_POINT_SMOOTH_HINT, GL_DONT_CARE);
}

void drawAll()
{
	if(updateDraw){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glFrontFace(GL_CCW);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
		glDepthFunc(GL_LEQUAL);
		glClearDepth(1.0);

		glPointSize(2);
		glLoadIdentity();

		oglcamera.Update();

//		glColor3f(0.0f,1.0f,0.0f);
//		glBegin(GL_QUADS);
//		double clip =10;
//		glVertex3f(clip,sys.groundHeight,clip);
//		glVertex3f(clip,sys.groundHeight,-clip);
//		glVertex3f(-clip,sys.groundHeight,-clip);
//		glVertex3f(-clip,sys.groundHeight,clip);
//		glEnd();
//		glFlush();

//		glColor3f(0.0f,0.0f,1.0f);
//		glPushMatrix();
//		float3 position = sys.getXYZPosition(100,0);
//		//cout << position.x << " " << position.y << " " << position.z << endl;
//		glTranslatef(position.x,position.y,position.z);
//		glutSolidSphere(1,10,10);
//		glPopMatrix();

		for (int i = 0; i < sys.particles.size(); i++) {
			glColor3f(0.0f, 1.0f, 0.0f);
			glPushMatrix();
			float3 pos = sys.getXYZPositionParticle(i);
			glTranslatef(pos.x, pos.y, pos.z);
			glutSolidSphere(sys.particles[i].getRadius(), 30, 30);
			glPopMatrix();

			//indicate velocity
			glLineWidth(sys.elements[i].getRadius()*500);
			glColor3f(1.0f,0.0f,0.0f);
			glBegin(GL_LINES);
			glVertex3f(pos.x,pos.y,pos.z);
			float3 vel = sys.getXYZVelocityParticle(i);
			//cout << "v:" << vel.x << " " << vel.y << " " << vel.z << endl;
			pos +=2*sys.particles[i].getRadius()*normalize(vel);
			glVertex3f(pos.x,pos.y,pos.z);
			glEnd();
			glFlush();
		}

		for(int i=0;i<sys.elements.size();i++)
		{
			int xiDiv = sys.numContactPoints;

			double xiInc = 1/(static_cast<double>(xiDiv-1));

			if(showSphere)
			{
				glColor3f(0.0f,0.0f,1.0f);
				for(int j=0;j<xiDiv;j++)
				{
					glPushMatrix();
					float3 position = sys.getXYZPosition(i,xiInc*j);
					glTranslatef(position.x,position.y,position.z);
					glutSolidSphere(sys.elements[i].getRadius(),10,10);
					glPopMatrix();
				}
			}
			else
			{
				int xiDiv = sys.numContactPoints;
				double xiInc = 1/(static_cast<double>(xiDiv-1));
				glLineWidth(sys.elements[i].getRadius()*500);
				glColor3f(0.0f,1.0f,0.0f);
				glBegin(GL_LINE_STRIP);
				for(int j=0;j<sys.numContactPoints;j++)
				{
					float3 position = sys.getXYZPosition(i,xiInc*j);
					glVertex3f(position.x,position.y,position.z);
				}
				glEnd();
				glFlush();
			}
		}

		glutSwapBuffers();
	}
}

void renderSceneAll(){
	if(OGL){
		//if(sys.timeIndex%10==0)
			drawAll();
		sys.DoTimeStep();
	}
}

void CallBackKeyboardFunc(unsigned char key, int x, int y) {
	switch (key) {
	case 'w':
		oglcamera.Forward();
		break;
	case 's':
		oglcamera.Back();
		break;

	case 'd':
		oglcamera.Right();
		break;

	case 'a':
		oglcamera.Left();
		break;

	case 'q':
		oglcamera.Up();
		break;

	case 'e':
		oglcamera.Down();
		break;
	}
}

void CallBackMouseFunc(int button, int state, int x, int y) {
	oglcamera.SetPos(button, state, x, y);
}
void CallBackMotionFunc(int x, int y) {
	oglcamera.Move2D(x, y);
}

int main(int argc, char** argv)
{
	sys.setTimeStep(1e-3);
	sys.setTolerance(1e-6);
	sys.useSpike = atoi(argv[1]);
	sys.numContactPoints = 30;
	sys.setPartitions(atoi(argv[2]));

	if(argc == 3)
	{
		Element test = Element();
		test.setElasticModulus(2e11);
		sys.addElement(&test);
		sys.addConstraint_AbsoluteSpherical(0);
		sys.numContactPoints = 10;
	}
	else
	{
		sys.fullJacobian = 1;
		double length = 1;
		double r = .02;
		double E = 2e6;
		double rho = 2200;
		double nu = .3;
		int numElementsPerSide = atoi(argv[3]);
		Element element;
		int k = 0;
		// Add elements in x-direction
		for (int j = 0; j < numElementsPerSide+1; j++) {
			for (int i = 0; i < numElementsPerSide; i++) {
				element = Element(Node(i*length, 0, j*length, 1, 0, 0),
								  Node((i+1)*length, 0, j*length, 1, 0, 0),
								  r, nu, E, rho);
				sys.addElement(&element);
				k++;
				if(k%100==0) printf("Elements %d\n",k);
			}
		}

		// Add elements in z-direction
		for (int j = 0; j < numElementsPerSide+1; j++) {
			for (int i = 0; i < numElementsPerSide; i++) {
				element = Element(Node(j*length, 0, i*length, 0, 0, 1),
								  Node(j*length, 0, (i+1)*length, 0, 0, 1),
								  r, nu, E, rho);
				sys.addElement(&element);
				k++;
				if(k%100==0) printf("Elements %d\n",k);
			}
		}

//		// Test Constraints
//		for(int j=0; j < numElementsPerSide+1; j++)
//		{
//			sys.addConstraint_AbsoluteSpherical(sys.elements[j * numElementsPerSide], 0);
//		}
//
//		for(int j=0; j < numElementsPerSide+1; j++)
//		{
//			sys.addConstraint_AbsoluteSpherical(sys.elements[j * numElementsPerSide+numElementsPerSide*(numElementsPerSide+1)], 0);
//		}
//		// End Test Constraints

		sys.addConstraint_AbsoluteSpherical(sys.elements[0], 0);
		sys.addConstraint_AbsoluteSpherical(sys.elements[2*numElementsPerSide*(numElementsPerSide+1)-numElementsPerSide], 0);
		sys.addConstraint_AbsoluteSpherical(sys.elements[numElementsPerSide*(numElementsPerSide+1)-numElementsPerSide], 0);
		sys.addConstraint_AbsoluteSpherical(sys.elements[2*numElementsPerSide*(numElementsPerSide+1)-1], 1);
		sys.addConstraint_AbsoluteSpherical(sys.elements[numElementsPerSide*(numElementsPerSide+1)-1], 1);


		// Constrain x-strands together
		for(int j=0; j < numElementsPerSide+1; j++)
		{
			for(int i=0; i < numElementsPerSide-1; i++)
			{
				sys.addConstraint_RelativeFixed(
						sys.elements[i+j*numElementsPerSide], 1,
						sys.elements[i+1+j*numElementsPerSide], 0);
			}
		}

		// Constrain z-strands together
		int offset = numElementsPerSide*(numElementsPerSide+1);
		for(int j=0; j < numElementsPerSide+1; j++)
		{
			for(int i=0; i < numElementsPerSide-1; i++)
			{
				sys.addConstraint_RelativeFixed(
						sys.elements[i+offset+j*numElementsPerSide], 1,
						sys.elements[i+offset+1+j*numElementsPerSide], 0);
			}
		}

		// Constrain cross-streams together
		for(int j=0; j < numElementsPerSide; j++)
		{
			for(int i=0; i < numElementsPerSide; i++)
			{
				sys.addConstraint_RelativeSpherical(
						sys.elements[i*numElementsPerSide+j], 0,
						sys.elements[offset+i+j*numElementsPerSide], 0);
			}
		}

		for(int i=0; i < numElementsPerSide; i++)
		{
			sys.addConstraint_RelativeSpherical(
						sys.elements[numElementsPerSide-1+numElementsPerSide*i], 1,
						sys.elements[2*offset-numElementsPerSide+i], 0);
		}

		for(int i=0; i < numElementsPerSide; i++)
		{
			sys.addConstraint_RelativeSpherical(
						sys.elements[numElementsPerSide*(numElementsPerSide+1)+numElementsPerSide-1+numElementsPerSide*i], 1,
						sys.elements[numElementsPerSide*numElementsPerSide+i], 0);
		}
	}



/*
	if (argc == 2) {
		sys.useSpike = atoi(argv[1]);
		Element test = Element();
		test.setElasticModulus(2e5);
		sys.addElement(&test);
		sys.addConstraint_AbsoluteSpherical(0);
		sys.numContactPoints = 100;
	} else {
		Particle particle1 = Particle(60,70,make_float3(60,70,60),make_float3(0,0,0));
		sys.addParticle(&particle1);
//
//		Particle particle2 = Particle(.6*100,100,make_float3(1200-60,2*100,60),make_float3(0,0,0));
//		sys.addParticle(&particle2);
//
//		Particle particle3 = Particle(.6*100,100,make_float3(1200-60,2*100,1200-60),make_float3(0,0,0));
//		sys.addParticle(&particle3);
//
//		Particle particle4 = Particle(.6*100,100,make_float3(60,2*100,1200-60),make_float3(0,0,0));
//		sys.addParticle(&particle4);

		//Horizontal Net
		sys.setTimeStep(1e-4);
		sys.setTolerance(1e-3);
		sys.detector.setBinsPerAxis(make_uint3(70,10,70));
		//sys.detector.activateDebugMode();
		sys.numContactPoints = 6;
		int numElements = atoi(argv[2]);
		double length = .3*100;
		//double EM = 2e7;
		//double rho = 1150.0;

		Particle particle;
		for( int i=5; i<(numElements-5)/2;i++) {
			for(int j=0;j<5;j++) {
				for(int k=5;k<(numElements-5)/2;k++) {
					particle = Particle(length,10,make_float3(length+2*length*i,3*length+3*length*j,length+2*length*k),make_float3(0,0,0));
					sys.addParticle(&particle);
				}
			}
		}

		Element element;
		int k = 0;
		for (int i = 0; i < numElements+1; i++) {
			for (int j = 0; j < numElements; j++) {
				element = Element(Node(i * length, 0, j * length, 0, 0, 1),
						Node(i * length, 0, (j + 1) * length, 0, 0, 1));
				//element.setElasticModulus(EM);
				//element.setDensity(rho);
				sys.addElement(&element);
				if (sys.elements.size() % 1000 == 0)
					printf("Elements added: %d\n", sys.elements.size());

				//printf("Element #%d: (%f, %f, %f) -> (%f, %f, %f)\n",k,i*length,-j*length,0.0,i*length,-(j+1)*length,0.0);
				k++;
			}
		}

		for (int i = 0; i < numElements+1; i++) {
			if(i==0||i==numElements)
			{
				sys.addConstraint_AbsoluteSpherical(sys.elements[i * numElements], 0);
				sys.addConstraint_AbsoluteSpherical(sys.elements[(i+1)*numElements-1], 1);
				//printf("Constraint: %d\n",numElements*i);
			}
		}

		for (int i = 0; i < numElements+1; i++) {
			for (int j = 0; j < numElements - 1; j++) {
				sys.addConstraint_RelativeFixed(
						sys.elements[j + numElements * i], 1,
						sys.elements[j + 1 + numElements * i], 0);
				//printf("Constraints: %d to %d\n",j+numElements*i,j+1+numElements*i);
			}
		}

		for (int i = 0; i < numElements; i++) {
			element = Element(
					Node(i * length, 0, 0, 1, 0, 0),
					Node((i + 1) * length, 0, 0, 1, 0, 0));
			//element.setElasticModulus(EM);
			//element.setDensity(rho);
			sys.addElement(&element);
			//if (sys.elements.size() % 1000 == 0)
				//printf("Elements added: %d\n", sys.elements.size());

			//printf("Cross Element #%d: (%f, %f, %f) -> (%f, %f, %f)\n",k,i*length,-(j+1)*length,0.0,(i+1)*length,-(j+1)*length,0.0);

			sys.addConstraint_RelativeSpherical(
					sys.elements[0 + numElements * i], 0,
					sys.elements[k], 0);
			sys.addConstraint_RelativeSpherical(
					sys.elements[0 + numElements * (i + 1)], 0,
					sys.elements[k], 1);
			if(i==0) sys.addConstraint_AbsoluteSpherical(sys.elements[k],0);
			if(i==numElements-2) sys.addConstraint_AbsoluteSpherical(sys.elements[k],1);
			//printf("Cross Constraint: %d to %d\n",j+numElements*i,k);
			//printf("Cross Constraint: %d to %d\n",j+numElements*(i+1),k);
			k++;
		}

		for (int i = 0; i < numElements; i++) {
			for (int j = 0; j < numElements; j++) {
				element = Element(
						Node(i * length, 0, (j + 1) * length, 1, 0, 0),
						Node((i + 1) * length, 0, (j + 1) * length, 1, 0, 0));
				//element.setElasticModulus(EM);
				//element.setDensity(rho);
				sys.addElement(&element);
				//if (sys.elements.size() % 1000 == 0)
					//printf("Elements added: %d\n", sys.elements.size());

				//printf("Cross Element #%d: (%f, %f, %f) -> (%f, %f, %f)\n",k,i*length,-(j+1)*length,0.0,(i+1)*length,-(j+1)*length,0.0);

				sys.addConstraint_RelativeSpherical(
						sys.elements[j + numElements * i], 1, sys.elements[k],
						0);
				sys.addConstraint_RelativeSpherical(
						sys.elements[j + numElements * (i + 1)], 1,
						sys.elements[k], 1);
				//printf("Cross Constraint: %d to %d\n",j+numElements*i,k);
				//printf("Cross Constraint: %d to %d\n",j+numElements*(i+1),k);
				if(i==0&&(j==0||j==numElements-1)) sys.addConstraint_AbsoluteSpherical(sys.elements[k],0);
				if(i==numElements-1&&(j==0||j==numElements-1)) sys.addConstraint_AbsoluteSpherical(sys.elements[k],1);
				k++;
			}
		}
	}

//	else
//	{
//		//SOCCER NET
//		sys.setTimeStep(1e-4);
//		sys.setTolerance(1e-8);
//		sys.numContactPoints = 10;
//		int numElements = atoi(argv[1]);
//		double length = .3;
//		double EM = 2e7;
//		double rho = 1150.0;
//
//		Element element;
//		int k = 0;
//		for(int i=0;i<numElements;i++)
//		{
//			for(int j=0;j<numElements;j++)
//			{
//				element = Element(Node(i*length,-j*length,0,0,-1,0),Node(i*length,-(j+1)*length,0,0,-1,0));
//				element.setElasticModulus(EM);
//				element.setDensity(rho);
//				sys.addElement(&element);
//				if(sys.elements.size()%1000==0) printf("Elements added: %d\n",sys.elements.size());
//
//				//printf("Element #%d: (%f, %f, %f) -> (%f, %f, %f)\n",k,i*length,-j*length,0.0,i*length,-(j+1)*length,0.0);
//				k++;
//			}
//		}
//
//		for(int i=0;i<numElements;i++)
//		{
//			sys.addConstraint_AbsoluteSpherical(sys.elements[i*numElements],0);
//			//sys.addConstraint_AbsoluteSpherical(sys.elements[(i+1)*numElements-1],1);
//			//printf("Constraint: %d\n",numElements*i);
//		}
//
//		for(int i=0;i<numElements;i++)
//		{
//			for(int j=0;j<numElements-1;j++)
//			{
//				sys.addConstraint_RelativeFixed(sys.elements[j+numElements*i],1,sys.elements[j+1+numElements*i],0);
//				//printf("Constraints: %d to %d\n",j+numElements*i,j+1+numElements*i);
//			}
//		}
//
//		for(int i=0;i<numElements-1;i++)
//		{
//			for(int j=0;j<numElements;j++)
//			{
//				element = Element(Node(i*length,-(j+1)*length,0,1,0,0),Node((i+1)*length,-(j+1)*length,0,1,0,0));
//				element.setElasticModulus(EM);
//				element.setDensity(rho);
//				sys.addElement(&element);
//				if(sys.elements.size()%1000==0) printf("Elements added: %d\n",sys.elements.size());
//
//				//printf("Cross Element #%d: (%f, %f, %f) -> (%f, %f, %f)\n",k,i*length,-(j+1)*length,0.0,(i+1)*length,-(j+1)*length,0.0);
//
//
//				sys.addConstraint_RelativeSpherical(sys.elements[j+numElements*i],1,sys.elements[k],0);
//				sys.addConstraint_RelativeSpherical(sys.elements[j+numElements*(i+1)],1,sys.elements[k],1);
//				//printf("Cross Constraint: %d to %d\n",j+numElements*i,k);
//				//printf("Cross Constraint: %d to %d\n",j+numElements*(i+1),k);
//				k++;
//			}
//		}
//	}
*/
	printf("Initializing system (%d beams, %d constraints)... ",sys.elements.size(),sys.constraints.size());
	sys.initializeSystem();
	printf("System Initialized (%d beams, %d constraints, %d equations)!\n",sys.elements.size(),sys.constraints.size(),12*sys.elements.size()+sys.constraints.size());

//	while(sys.timeIndex<=30)
//	{
//		//if(sys.getTimeIndex()%100==0) sys.writeToFile();
//		sys.DoTimeStep();
//	}
//	printf("Total time to simulate: %f [s]\n",sys.timeToSimulate);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0,0);
	glutInitWindowSize(1024	,512);
	glutCreateWindow("MAIN");
	glutDisplayFunc(renderSceneAll);
	glutIdleFunc(renderSceneAll);
	glutReshapeFunc(changeSize);
	glutIgnoreKeyRepeat(0);
	glutKeyboardFunc(CallBackKeyboardFunc);
	glutMouseFunc(CallBackMouseFunc);
	glutMotionFunc(CallBackMotionFunc);
	initScene();
	glutMainLoop();


//#pragma omp parallel sections
//	{
//#pragma omp section
//		{
//			while(true)
//			{
////				sys.clearAppliedForces();
////				force.x = -forceMag*sys.p_h[element.getElementIndex()*12+10];
////				force.y = forceMag*sys.p_h[element.getElementIndex()*12+9];
////				force.z = 0;
////				sys.addForce(&element,1,force);
//				sys.DoTimeStep();
//				//if(sys.timeIndex%100==0) sys.writeToFile();
//			}
//		}
//#pragma omp section
//		{
//			if(OGL){
//				glutInit(&argc, argv);
//				glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
//				glutInitWindowPosition(0,0);
//				glutInitWindowSize(1024	,512);
//				glutCreateWindow("MAIN");
//				glutDisplayFunc(renderSceneAll);
//				glutIdleFunc(renderSceneAll);
//				glutReshapeFunc(changeSize);
//				glutIgnoreKeyRepeat(0);
//				glutKeyboardFunc(CallBackKeyboardFunc);
//				glutMouseFunc(CallBackMouseFunc);
//				glutMotionFunc(CallBackMotionFunc);
//				initScene();
//				glutMainLoop();
//			}
//		}
//	}

	return 0;
}

