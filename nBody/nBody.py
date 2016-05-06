import numpy as np
import sys, time, os
import h5py as h5
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel


currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
pointsDirectory = parentDirectory + "/points_visualization"
sys.path.extend( [toolsDirectory, pointsDirectory] )
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo
from tools import printProgressTime, timeSplit
from dataAnalysis import *


nParticles = 16000
#nParticles = 1024*64
totalSteps = 1000

#G    = 6.67384e-11 #m**2/(kg*s**2)
#mSun = 1.98e30     #kg
#pc   = 3.08e16     #m
#initialR =  1*pc #*np.random.random( nParticles ).astype(cudaPre)

G    = 1 #m**2/(kg*s**2)
mSun = 1     #kg
pMass = 1
initialR =  500

dt = 0.1
epsilon = .05

cudaP = "double"
devN = None
usingAnimation = False
showKernelMemInfo = False
plotting = True

#Read in-line parameters
for option in sys.argv:
  if option.find("part")>=0 : nParticles = int(option[option.find("=")+1:])
  if option.find("anim")>=0: usingAnimation = True
  if option.find("mem") >=0: showKernelMemInfo = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])

precision  = {"float":np.float32, "double":np.float64}
cudaPre = precision[cudaP]

pMass = cudaPre( pMass )
G = cudaPre( G )
GMass = G*pMass


#Initialize file for saving data
energyList = []
outputDir = '/home/bruno/data/nBody/'
if not os.path.exists( outputDir ): os.makedirs( outputDir )
outDataFile = outputDir + 'test_all.h5'
outFile = h5.File( outDataFile , "w")
posHD = outFile.create_group("pos")
energyHD = outFile.create_group("energy")




#Set CUDA thread grid dimentions
block = ( 160, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )


if usingAnimation:
  import points3D as pAnim #points3D Animation
  pAnim.nPoints = nParticles
  N = 3
  pAnim.viewXmin, pAnim.viewXmax = -N*initialR, N*initialR
  pAnim.viewYmin, pAnim.viewYmax = -N*initialR, N*initialR
  pAnim.viewZmin, pAnim.viewZmax = -N*initialR, N*initialR
  pAnim.showGrid = False
  pAnim.windowTitle = " CUDA N-body simulation     particles={0}".format(nParticles)

###########################################################################
###########################################################################
#Initialize and select CUDA device
if usingAnimation:
  pAnim.initGL()
  pAnim.CUDA_initialized = True
  #configAnimation()
cudaDev = setCudaDevice( devN = devN, usingAnimation = usingAnimation )

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
codeFiles = [ "vector3D.h", "cudaNbody.cu"]
for fileName in codeFiles:
  codeString = open(fileName, "r").read().replace("cudaP", cudaP)
  outFile = open( fileName + "T", "w" )
  outFile.write( codeString )
  outFile.close()
cudaCodeStringTemp = open("cudaNbody.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "TPB":block[0], "gDIM":grid[0], 'bDIM':block[0] }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
getAcccelKernel = cudaCode.get_function("getAccel_kernel" )
getEnergyKernel = cudaCode.get_function("getEnergy_kernel" )
if showKernelMemInfo:
  kernelMemoryInfo(mainKernel, 'mainKernel')
  sys.exit()
########################################################################
moveParticles = ElementwiseKernel(arguments="cudaP dt, cudaP *posX, cudaP *posY, cudaP *posZ,\
				     cudaP *velX, cudaP *velY, cudaP *velZ,\
				     cudaP *accelX, cudaP *accelY, cudaP *accelZ".replace( "cudaP", cudaP ),
			      operation = "posX[i] = posX[i] + dt*( velX[i] + 0.5*dt*accelX[i]);\
				           posY[i] = posY[i] + dt*( velY[i] + 0.5*dt*accelY[i]);\
				           posZ[i] = posZ[i] + dt*( velZ[i] + 0.5*dt*accelZ[i]);",
			      name ="moveParticles")
########################################################################
def getEnergy( step, time, energyList ):
  getEnergyKernel( np.int32( nParticles ), G, pMass,
		posX_d, posY_d, posZ_d, posX_d, posY_d, posZ_d,
		velX_d, velY_d, velZ_d, energyAll_d,
		cudaPre(epsilon), np.int32(1), grid=grid, block=block )
  energyAll_h = energyAll_d.get()
  energyList.append( [ step, time, energyAll_h.sum() ] )



########################################################################
def loadState( files=["galaxy.hdf5"] ):
  posAll = []
  velAll = []
  for fileName in files:
    dataFile = h5.File( currentDirectory+ "/" + fileName ,'r')
    pos = dataFile.get("posParticles")[...]
    vel = dataFile.get("velParticles")[...]
    print '\nLoading data... \n file: {0} \n particles: {1}\n'.format(fileName, pos.shape[1] )
    posAll.append( pos )
    velAll.append( vel )
    dataFile.close()
  return posAll, velAll

########################################################################
def saveState():
  dataFileName = "galaxy.hdf5"
  dataFile = h5.File(dataFileName,'w')
  dataFile.create_dataset( "posParticles", data=np.array([posX_d.get(), posY_d.get(), posZ_d.get() ]), compression='lzf')
  dataFile.create_dataset( "velParticles", data=np.array([velX_d.get(), velY_d.get(), velZ_d.get() ]), compression='lzf')
  dataFile.close()
  print "Data Saved: ", dataFileName, "\n"
########################################################################
def getRadialDist():
  posX_h, posY_h, posZ_h = posX_d.get(), posY_d.get(), posZ_d.get()
  cm = np.array([ posX_h.mean(), posY_h.mean(), posZ_h.mean() ])
########################################################################
#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )
#Spherically uniform random distribution for initial positions
initialTheta = 2*np.pi*np.random.rand(nParticles)
initialPhi = np.arccos(2*np.random.rand(nParticles) - 1)
##initialR = ( 50*pc )**3*np.random.random( nParticles )
##initialR = np.power(initialR, 1./3)*np.random.random( nParticles )
posX_h = initialR*np.cos(initialTheta)*np.sin(initialPhi)
posY_h = initialR*np.sin(initialTheta)*np.sin(initialPhi)
posZ_h = initialR*np.cos(initialPhi)

#posX_h[:nParticles/2] += 5000
#posX_h[nParticles/2:] -= 5000
##Spherically uniform random distribution for initial velocity
#initialTheta = 2*np.pi*np.random.rand(nParticles)
#initialPhi = np.arccos(2*np.random.rand(nParticles) - 1)
#initialR = 0.
#velX_h = initialR*np.cos(initialTheta)*np.sin(initialPhi)
#velY_h = initialR*np.sin(initialTheta)*np.sin(initialPhi)
#velZ_h = initialR*np.cos(initialPhi)
#######################
velX_h = np.zeros(nParticles)
velY_h = np.zeros(nParticles)
velZ_h = np.zeros(nParticles)
##################################################################
accelX_h = np.zeros(nParticles)
accelY_h = np.zeros(nParticles)
accelZ_h = np.zeros(nParticles)###########################################

#Disk Distribution
initialRAll = initialR**2 * np.random.random( nParticles )
initialRAll = np.sqrt( initialRAll )
initialTheta = np.linspace( 0, 2*np.pi, nParticles, endpoint=False)
posX_h = initialRAll*np.cos(initialTheta)
posY_h = initialRAll*np.sin(initialTheta)
posZ_h = 0*np.random.rand(nParticles)
posZ_h = np.zeros(nParticles)
velX_h = -2.9*posY_h/initialRAll
velY_h =  2.9*posX_h/initialRAll

shufle = np.random.randint(0,2,nParticles).astype(np.bool)
g1 = shufle
g2 = np.logical_not( shufle )

posZ_h[ g2 ] = posY_h[ g2 ]
posY_h[ g2 ] -= posY_h[ g2 ]
velZ_h[ g2 ] = velY_h[ g2 ]
velY_h[ g2 ] -= velY_h[ g2 ]

posX_h[ g1] += initialR/3.
posX_h[ g2 ] -= initialR/3.

posY_h[ g1] += initialR
posY_h[ g2 ] -= initialR

initialVel = 5
velY_h[ g1] -= initialVel
velY_h[ g2 ] += initialVel

pAnim.pointsColor = np.zeros(nParticles*3).astype(np.float32)
pAnim.pointsColor[::3][ g1 ] = 1
pAnim.pointsColor[2::3][ g2 ] = 1
pAnim.pointsColor = pAnim.pointsColor.astype(np.float32)



##Ring distribution
#dTheta = 2*np.pi / nParticles
#theta = 0.
#for i in range(nParticles):
  #posX_h[i] = initialR * np.cos( theta )
  #posY_h[i] = initialR * np.sin( theta )
  #posZ_h[i] = 0
  #theta += dTheta






#Load initial conditions from file
#nombre = 'Nbody_condiciones_iniciales.dat'

#datos = np.loadtxt( nombre )
#masa = datos[:,0]
#pos = datos[:,1:4].T
#vel = datos[:,4:7].T
#accel = datos[:,7:10].T
#energ = datos[0,10]

#posX_h, posY_h, posZ_h = pos[0].copy(), pos[1].copy(), pos[2].copy()
#velX_h, velY_h, velZ_h = vel[0].copy(), vel[1].copy(), vel[2].copy()
#accelX_h, accelY_h, accelZ_h = accel[0].copy(), accel[1].copy(), accel[2].copy()

#pMass = cudaPre( masa[2] )
#G = cudaPre( G )
#GMass = G*pMass

pos_h = ( np.concatenate([ posX_h, posY_h, posZ_h ]) ).astype(cudaPre)
posX_d = gpuarray.to_gpu( posX_h.astype(cudaPre) )
posY_d = gpuarray.to_gpu( posY_h.astype(cudaPre) )
posZ_d = gpuarray.to_gpu( posZ_h.astype(cudaPre) )
pos_d  = gpuarray.to_gpu( pos_h )
velX_d = gpuarray.to_gpu( velX_h.astype(cudaPre) )
velY_d = gpuarray.to_gpu( velY_h.astype(cudaPre) )
velZ_d = gpuarray.to_gpu( velZ_h.astype(cudaPre) )
accelX_d = gpuarray.to_gpu( accelX_h.astype(cudaPre))
accelY_d = gpuarray.to_gpu( accelY_h.astype(cudaPre) )
accelZ_d = gpuarray.to_gpu( accelZ_h.astype(cudaPre) )
energyAll_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
#Get initial accel
getAcccelKernel( np.int32(nParticles),  np.int32(0),
	    posX_d, posY_d, posZ_d,
	    velX_d, velY_d, velZ_d,
	    accelX_d, accelY_d, accelZ_d,
	    cudaPre( 0. ), cudaPre(epsilon),
	    np.intp(0),   grid=grid, block=block )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6)


def animationUpdate():
  global nAnimIter, runningTime
  start, end = cuda.Event(), cuda.Event()
  start.record()
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  getAcccelKernel( np.int32(nParticles), np.int32(usingAnimation),
	      posX_d, posY_d, posZ_d,
	      velX_d, velY_d, velZ_d,
	      accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon),
	      np.intp(pAnim.cuda_VOB_ptr),   grid=grid, block=block )
  nAnimIter += 1
  #end.record()
  #end.synchronize()
  #secs = start.time_till(end)*1e-3
  #runningTime += secs
  #if nAnimIter == 50:
    #print 'Steps per sec: {0:0.2f}'.format( 50/runningTime  )
    #nAnimIter, runningTime = 0, 0

def keyboard(*args):
  global viewXmin, viewXmax, viewYmin, viewYmax
  global showGrid, gridCenter
  ESCAPE = '\033'
  if args[0] == ESCAPE:
    print "Ending Simulation"
    sys.exit()
  elif args[0] == "s":
    saveState()
###########################################################################
###########################################################################

nAnimIter = 0
runningTime = 0
simulationTime = 0

#Start Simulation
if plotting: plt.ion(), plt.show()
print "\nStarting simulation"
print " Using {0} precision".format( cudaP )
print ' nParticles: ', nParticles


if usingAnimation:
  pAnim.updateFunc = animationUpdate
  pAnim.keyboard = keyboard
  pAnim.startAnimation()







print ' nSteps: {0} \n'.format( totalSteps )
print " Output: {0} \n".format( outDataFile )
start, end = cuda.Event(), cuda.Event()
for stepCounter in range(totalSteps):
  start.record()
  if stepCounter%1 == 0: printProgressTime( stepCounter, totalSteps,  runningTime )
  getEnergy( stepCounter, simulationTime, energyList )
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  getAcccelKernel( np.int32(nParticles), GMass, np.int32(usingAnimation),
	      posX_d, posY_d, posZ_d, posX_d, posY_d, posZ_d,
	      velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon), np.int32(1),
	      np.intp(0),   grid=grid, block=block )
  simulationTime += dt
  end.record()
  end.synchronize()
  runningTime += start.time_till(end)*1e-3



h, m, s = timeSplit( runningTime )
print '\n\nTotal time: {0}:{1:02}:{2:02} '.format( h, m, s )

#Write energy
energyList = np.array( energyList )
energyHD.create_dataset( 'steps', data= energyList[:,0], compression='lzf' )
energyHD.create_dataset( 'time', data= energyList[:,1], compression='lzf' )
energyHD.create_dataset( 'values', data= energyList[:,2], compression='lzf' )

outFile.close()
