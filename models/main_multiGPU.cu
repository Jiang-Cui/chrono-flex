#include "include.cuh"
#include "ANCFSystem.cuh"
#include "Element.cuh"
#include "Node.cuh"
#include "Particle.cuh"

bool updateDraw = 1;
bool showSphere = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
const int numSystems = 2;
ANCFSystem* sys[numSystems];

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(-1,1,-1),camreal3(0,0,0),camreal3(0,1,0),.01);

// OPENGL RENDERING CODE //
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

    for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) {
      for (int i = 0; i < sys[sysIndex]->particles.size(); i++) {
        glColor3f(0.0f, 1.0f, 0.0f);
        glPushMatrix();
        float3 pos = sys[sysIndex]->getXYZPositionParticle(i);
        glTranslatef(pos.x, pos.y, pos.z);
        glutSolidSphere(sys[sysIndex]->particles[i].getRadius(), 30, 30);
        glPopMatrix();

        //indicate velocity
        glLineWidth(sys[sysIndex]->elements[i].getRadius()*500);
        glColor3f(1.0f,0.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(pos.x,pos.y,pos.z);
        float3 vel = sys[sysIndex]->getXYZVelocityParticle(i);
        //cout << "v:" << vel.x << " " << vel.y << " " << vel.z << endl;
        pos +=2*sys[sysIndex]->particles[i].getRadius()*normalize(vel);
        glVertex3f(pos.x,pos.y,pos.z);
        glEnd();
        glFlush();
      }

      for(int i=0;i<sys[sysIndex]->elements.size();i++)
      {
        int xiDiv = sys[sysIndex]->numContactPoints;

        double xiInc = 1/(static_cast<double>(xiDiv-1));

        if(showSphere)
        {
          glColor3f(0.0f,0.0f,1.0f);
          for(int j=0;j<xiDiv;j++)
          {
            glPushMatrix();
            float3 position = sys[sysIndex]->getXYZPosition(i,xiInc*j);
            glTranslatef(position.x,position.y,position.z);
            glutSolidSphere(sys[sysIndex]->elements[i].getRadius(),10,10);
            glPopMatrix();
          }
        }
        else
        {
          int xiDiv = sys[sysIndex]->numContactPoints;
          double xiInc = 1/(static_cast<double>(xiDiv-1));
          glLineWidth(sys[sysIndex]->elements[i].getRadius()*500);
          glColor3f(0.0f,1.0f,0.0f);
          glBegin(GL_LINE_STRIP);
          for(int j=0;j<sys[sysIndex]->numContactPoints;j++)
          {
            float3 position = sys[sysIndex]->getXYZPosition(i,xiInc*j);
            glVertex3f(position.x,position.y,position.z);
          }
          glEnd();
          glFlush();
        }
      }
    }
    glutSwapBuffers();
  }
}

void renderSceneAll(){
  if(OGL){
    //if(sys->timeIndex%10==0)
    drawAll();
    for(int i=0; i < numSystems; i++) {
      // Force a preconditioner update if needed
      if ((sys[i]->preconditionerUpdateModulus > 0) && (sys[i]->timeIndex % sys[i]->preconditionerUpdateModulus == 0)) {
        //mySolver->update(lhs.values);
        delete sys[i]->mySolver;
        sys[i]->mySolver = new SpikeSolver(sys[i]->partitions, sys[i]->solverOptions);
        sys[i]->mySolver->setup(sys[i]->lhs);
        sys[i]->precUpdated = true;
        printf("Preconditioner updated (step condition)!\n");
      }

      sys[i]->DoTimeStep();
    }
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
#endif
// END OPENGL RENDERING CODE //

int main(int argc, char** argv)
{
  /////////////////////////////////////////////////////////////////////////
  //
  // Set up the system
  //
  ////////////////////////////////////////////////////////////////////////

  // command line arguments
  // ImplicitBeamsGPU <numPartitions> <numBeamsPerSide> <solverType> <usePreconditioning> <elasticModulus> <dataFolder>
  // solverType: (0) BiCGStab, (1) BiCGStab1, (2) BiCGStab2, (3) MinRes

#ifdef WITH_GLUT
  bool visualize = true;
#endif

  int numElementsPerSide = 4;
  double E = 2e7;
  double t_end = 5.0;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int    outputInterval = 100;
  double length = 1;
  double r = .02;
  double rho = 2200;
  double nu = .3;
  string data_folder = "./garbage";

  for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) {
    sys[sysIndex] = new ANCFSystem();

    sys[sysIndex]->setTimeStep(1e-3, 1e-10);
    sys[sysIndex]->setMaxNewtonIterations(20);
    sys[sysIndex]->setMaxKrylovIterations(5000);
    sys[sysIndex]->numContactPoints = 30;

    sys[sysIndex]->setNumPartitions(1);

    sys[sysIndex]->setSolverType(2);
    sys[sysIndex]->setPrecondType(0);
    sys[sysIndex]->fullJacobian = 1;


    if(argc>1) {
      sys[sysIndex]->setNumPartitions((int)atoi(argv[1]));
      numElementsPerSide = atoi(argv[2]);
      sys[sysIndex]->setSolverType((int)atoi(argv[3]));
      sys[sysIndex]->setPrecondType(atoi(argv[4]));
      if(atoi(argv[4])) {
        sys[sysIndex]->preconditionerUpdateModulus = precUpdateInterval;
        sys[sysIndex]->preconditionerMaxKrylovIterations = precMaxKrylov;
      }
      E = atof(argv[5]);
      data_folder = argv[6];
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Add elements to system
  //
  ////////////////////////////////////////////////////////////////////////

  Element element;
  int k = 0;
  // Add elements in x-direction
  for (int j = 0; j < numElementsPerSide+1; j++) {
    for (int i = 0; i < numElementsPerSide; i++) {
      element = Element(Node(i*length, 0, j*length, 1, 0, 0),
          Node((i+1)*length, 0, j*length, 1, 0, 0),
          r, nu, E, rho);
      for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addElement(&element);
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
      for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addElement(&element);
      k++;
      if(k%100==0) printf("Elements %d\n",k);
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Add constraints to system
  //
  ////////////////////////////////////////////////////////////////////////

  // Fix corners to ground
  for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) {
    sys[sysIndex]->addConstraint_AbsoluteSpherical(sys[sysIndex]->elements[0], 0);
    sys[sysIndex]->addConstraint_AbsoluteSpherical(sys[sysIndex]->elements[2*numElementsPerSide*(numElementsPerSide+1)-numElementsPerSide], 0);
    sys[sysIndex]->addConstraint_AbsoluteSpherical(sys[sysIndex]->elements[numElementsPerSide*(numElementsPerSide+1)-numElementsPerSide], 0);
    sys[sysIndex]->addConstraint_AbsoluteSpherical(sys[sysIndex]->elements[2*numElementsPerSide*(numElementsPerSide+1)-1], 1);
    sys[sysIndex]->addConstraint_AbsoluteSpherical(sys[sysIndex]->elements[numElementsPerSide*(numElementsPerSide+1)-1], 1);
  }

  // Constrain x-strands together
  for(int j=0; j < numElementsPerSide+1; j++)
  {
    for(int i=0; i < numElementsPerSide-1; i++)
    {
      for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addConstraint_RelativeFixed(
          sys[sysIndex]->elements[i+j*numElementsPerSide], 1,
          sys[sysIndex]->elements[i+1+j*numElementsPerSide], 0);
    }
  }

  // Constrain z-strands together
  int offset = numElementsPerSide*(numElementsPerSide+1);
  for(int j=0; j < numElementsPerSide+1; j++)
  {
    for(int i=0; i < numElementsPerSide-1; i++)
    {
      for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addConstraint_RelativeFixed(
          sys[sysIndex]->elements[i+offset+j*numElementsPerSide], 1,
          sys[sysIndex]->elements[i+offset+1+j*numElementsPerSide], 0);
    }
  }

  // Constrain cross-streams together
  for(int j=0; j < numElementsPerSide; j++)
  {
    for(int i=0; i < numElementsPerSide; i++)
    {
      for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addConstraint_RelativeSpherical(
          sys[sysIndex]->elements[i*numElementsPerSide+j], 0,
          sys[sysIndex]->elements[offset+i+j*numElementsPerSide], 0);
    }
  }

  for(int i=0; i < numElementsPerSide; i++)
  {
    for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addConstraint_RelativeSpherical(
        sys[sysIndex]->elements[numElementsPerSide-1+numElementsPerSide*i], 1,
        sys[sysIndex]->elements[2*offset-numElementsPerSide+i], 0);
  }

  for(int i=0; i < numElementsPerSide; i++)
  {
    for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) sys[sysIndex]->addConstraint_RelativeSpherical(
        sys[sysIndex]->elements[numElementsPerSide*(numElementsPerSide+1)+numElementsPerSide-1+numElementsPerSide*i], 1,
        sys[sysIndex]->elements[numElementsPerSide*numElementsPerSide+i], 0);
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Finalize the system
  //
  ////////////////////////////////////////////////////////////////////////
  for(int sysIndex = 0; sysIndex < numSystems; sysIndex++) {
    printf("%d, %d, %d\n",sys[sysIndex]->elements.size(),sys[sysIndex]->constraints.size(),12*sys[sysIndex]->elements.size()+sys[sysIndex]->constraints.size());
    sys[sysIndex]->initializeSystem();
    printf("System[%d] initialized!\n",sysIndex);
    sys[sysIndex]->printSolverParams();
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Perform the simulation
  //
  ////////////////////////////////////////////////////////////////////////

#ifdef WITH_GLUT
  if(visualize)
  {
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
  }
#endif
/*
  stringstream ss_m;
  ss_m << data_folder << "/" << "timing_" << atoi(argv[1]) << "_" << atoi(argv[2]) << "_" << atoi(argv[3]) << "_" << atoi(argv[4]) << "_" << atof(argv[5]) << ".txt";
  string timing_file_name = ss_m.str();
  ofstream ofile(timing_file_name.c_str());

  // if you don't want to visualize, then output the data
  int fileIndex = 0;
  while(sys->time < t_end)
  {
    if(sys->getTimeIndex()%outputInterval==0)
    {
      stringstream ss;
      //cout << "Frame: " << fileIndex << endl;
      ss << data_folder << "/" << fileIndex << ".txt";
      sys->writeToFile(ss.str());
      fileIndex++;
    }

    // Force a preconditioner update if needed
    if ((sys->preconditionerUpdateModulus > 0) && (sys->timeIndex % sys->preconditionerUpdateModulus == 0)) {
      //mySolver->update(lhs.values);
      delete sys->mySolver;
      sys->mySolver = new SpikeSolver(sys->partitions, sys->solverOptions);
      sys->mySolver->setup(sys->lhs);
      sys->precUpdated = true;
      printf("Preconditioner updated (step condition)!\n");
    }

    sys->DoTimeStep();
    ofile << sys->time                 << ", "
        << sys->stepTime             << ", "
        << sys->stepNewtonIterations << ", "
        << sys->stepKrylovIterations << ", "
        << sys->precUpdated          << " ,     ";
    for (size_t i = 0; i < sys->stepNewtonIterations; ++i)
      ofile << sys->spikeSolveTime[i] << ", " << sys->spikeNumIter[i] << ",     ";
    ofile << endl;
  }
  printf("Total time to simulate: %f [s]\n",sys->timeToSimulate);
  ofile.close();
*/
  return 0;
}

