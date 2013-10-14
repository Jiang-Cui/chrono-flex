#include "include.cuh"
#include "ANCFSystem.cuh"
#include "Element.cuh"
#include "Node.cuh"
#include "Particle.cuh"

bool updateDraw = 1;
bool showSphere = 1;

ANCFSystem sys;
OpenGLCamera oglcamera(camreal3(-1,1,-1),camreal3(0,0,0),camreal3(0,1,0),.01);

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
		test.setElasticModulus(2e7);
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

		// Fix corners to ground
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

	printf("Initializing system (%d beams, %d constraints)... ",sys.elements.size(),sys.constraints.size());
	sys.initializeSystem();
	printf("System Initialized (%d beams, %d constraints, %d equations)!\n",sys.elements.size(),sys.constraints.size(),12*sys.elements.size()+sys.constraints.size());

////	// Uncomment if you don't want visualization
//	while(sys.timeIndex<=30)
//	{
//		//if(sys.getTimeIndex()%100==0) sys.writeToFile();
//		sys.DoTimeStep();
//	}
//	printf("Total time to simulate: %f [s]\n",sys.timeToSimulate);

	// Uncomment if you want visualization
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

	return 0;
}

