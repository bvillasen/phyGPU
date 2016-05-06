import numpy as np
import sys, time, os, inspect, datetime
import h5py as h5
from mpi4py import MPI
#import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
sys.path.append( toolsDirectory )
from cudaTools import *
from tools import printProgressTime, timeSplit
from mpiTools import transferData

nParticles = 1024*32
#nParticles = 1024*128
nParticles = 1024*256
#nParticles = 1024*512

totalSteps = 100

G    = 1     #m**2/(kg*s**2)
mSun = 3     #kg
initialR =  5000

dt = 5
epsilon = 5

cudaP = "double"
#Get in-line parameters
for option in sys.argv:
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]
GMass = cudaPre( G*mSun ) 

#Initialize MPI
MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()
name = MPI.Get_processor_name()

pLeft  = nProc-1 if pId==0   else pId-1
pRight = 0 if pId==(nProc-1) else pId+1
nTotalParticles = nProc * nParticles

if pId == 0:
  print "\nMPI-CUDA nBoby"
  print " nProcess: {0}\n".format(nProc) 
MPIcomm.Barrier()
print "[pId {0}] Host: {1}".format( pId, name )

#Initialize file for saving data
outputDir = '/home_local/bruno/data/nBody/'
if not os.path.exists( outputDir ): os.makedirs( outputDir )
outDataFile = outputDir + 'test_{0}.h5'.format(pId)
dFile = h5.File( outDataFile , "w")
posHD = dFile.create_group("pos")

#Initialize CUDA
cudaCtx, cudaDev = mpi_setCudaDevice(pId, 0, MPIcomm, show=False)
#Set CUDA thread grid dimentions
block = ( 160, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )
MPIcomm.Barrier()
time.sleep(1)

#Read and compile CUDA code
if pId == 0: 
  print "\nCompiling CUDA code\n"
  codeFiles = [ "vector3D.h", "cudaNbody.cu"]
  for fileName in codeFiles:
    codeString = open(fileName, "r").read().replace("cudaP", cudaP)
    outFile = open( fileName + "T", "w" )
    outFile.write( codeString )
    outFile.close()
    time.sleep(1)
MPIcomm.Barrier()
cudaCodeStringTemp = open("cudaNbody.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "TPB":block[0], "gDIM":grid[0], 'bDIM':block[0] }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
mainKernel = cudaCode.get_function("main_kernel" )
##################################################################
moveParticles = ElementwiseKernel(arguments="cudaP dt, cudaP *posX, cudaP *posY, cudaP *posZ,\
				     cudaP *velX, cudaP *velY, cudaP *velZ,\
				     cudaP *accelX, cudaP *accelY, cudaP *accelZ".replace( "cudaP", cudaP ),
			      operation = "posX[i] = posX[i] + dt*( velX[i] + 0.5f*dt*accelX[i]);\
				           posY[i] = posY[i] + dt*( velY[i] + 0.5f*dt*accelY[i]);\
				           posZ[i] = posZ[i] + dt*( velZ[i] + 0.5f*dt*accelZ[i]);",
			      name ="moveParticles")
##################################################################
MPIcomm.Barrier()
########################################################################
#Initialize all gpu data
if pId == 0: 
  print "Initializing Data"
  initialMemory = getFreeMemory( show=True )  
##Spherically uniform random distribution for initial positions
initialTheta = 2*np.pi*np.random.rand(nParticles)
initialPhi = np.arccos(2*np.random.rand(nParticles) - 1) 
posX_h = ( initialR*np.cos(initialTheta)*np.sin(initialPhi)  ).astype(cudaPre)
posY_h = ( initialR*np.sin(initialTheta)*np.sin(initialPhi)  ).astype(cudaPre)
posZ_h = ( initialR*np.cos(initialPhi) ).astype(cudaPre)
offset = np.array([ [ 1, -1 ],  [ 1, 1 ], [ -1, 1 ], [ -1, -1 ] ]) * 1.1*initialR
posX_h += offset[pId][0]
posY_h += offset[pId][1]
#Arrays for data transfers
posSend_h = np.zeros( [nParticles, 3]      , dtype=cudaPre )
posAll_h  = np.zeros( [nTotalParticles, 3] , dtype=cudaPre )
posSend_h[:,0], posSend_h[:,1], posSend_h[:,2] = posX_h, posY_h, posZ_h
#Transfer intial data across all gpus
MPIcomm.Allgather( [posSend_h, MPI.DOUBLE], [posAll_h, MPI.DOUBLE] )
posAllX_h = posAll_h[:,0].copy()
posAllY_h = posAll_h[:,1].copy()
posAllZ_h = posAll_h[:,2].copy()
#Inital velocities
initialR = 0.
velX_h = initialR*np.cos(initialTheta)*np.sin(initialPhi)
velY_h = initialR*np.sin(initialTheta)*np.sin(initialPhi)
velZ_h = initialR*np.cos(initialPhi)
##################################################################
posX_d = gpuarray.to_gpu( posX_h )
posY_d = gpuarray.to_gpu( posY_h )
posZ_d = gpuarray.to_gpu( posZ_h )
posAllX_d = gpuarray.to_gpu( posAllX_h )
posAllY_d = gpuarray.to_gpu( posAllY_h )
posAllZ_d = gpuarray.to_gpu( posAllZ_h )
velX_d = gpuarray.to_gpu( velX_h.astype(cudaPre) )
velY_d = gpuarray.to_gpu( velY_h.astype(cudaPre) )
velZ_d = gpuarray.to_gpu( velZ_h.astype(cudaPre) )
accelX_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelY_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelZ_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
if pId == 0: 
  finalMemory = getFreeMemory( show=False )
  print " Global memory used per GPU: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 
MPIcomm.Barrier()
time.sleep(1)
print "[pId {0}] Output file: {1}".format( pId, outDataFile )
MPIcomm.Barrier()
time.sleep(1)

def sendPositions():
  global posSend_h, posAll_h, posX_h, posY_h, posZ_h
  posSend_h[:,0], posSend_h[:,1], posSend_h[:,2] = posX_d.get(), posY_d.get(), posZ_d.get() 
  MPIcomm.Barrier()
  MPIcomm.Allgather( [posSend_h, MPI.DOUBLE], [posAll_h, MPI.DOUBLE] )
  posAllX_h = posAll_h[:,0].copy()
  posAllY_h = posAll_h[:,1].copy()
  posAllZ_h = posAll_h[:,2].copy()
  posAllX_d.set( posAllX_h )
  posAllY_d.set( posAllY_h )
  posAllZ_d.set( posAllZ_h )


#def sendPositions():
  #global posSend_h, posRecv_h
  #posAll_h[0], posAll_h[1], posAll_h[2] = posX_d.get(), posY_d.get(), posZ_d.get() 
  #transferData( pId, pRight, pLeft, posSend_h, posRecv_h, step,  MPIcomm, block=False )
  #posX_d.set( posRecv_h[0] )
  #posY_d.set( posRecv_h[1] )
  #posZ_d.set( posRecv_h[2] )
  #posSend_h, posRecv_h, posRecv_h, posSend_h

def timeStep():
  global transferTime, computeTime
  #Transfer positions
  start.record()
  sendPositions()
  end.record(), end.synchronize()
  transferTime += start.time_till(end)*1e-3
  #Compute forces with all data  
  start.record()
  mainKernel( np.int32(nParticles), GMass, np.int32(0), 
	      posX_d, posY_d, posZ_d, posAllX_d, posAllY_d, posAllZ_d,
	      velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon), np.int32(nProc),
	      np.int32(0), grid=grid, block=block )
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  end.record(), end.synchronize()
  computeTime += start.time_till(end)*1e-3

###########################################################################
###########################################################################
#Start Simulation
if pId == 0:
  print "\nStarting simulation"
  print " Using {0} precision".format( cudaP )
  print " Total particles: ", nParticles*nProc
  print ' Particles per gpu: ', nParticles
  print ' nSteps: ', totalSteps
  print ''
MPIcomm.Barrier()


computeTime  = 0
transferTime = 0
saveTime = 0
step = 0
start, end = cuda.Event(), cuda.Event()
startT = cuda.Event()
startT.record()
for stepCounter in range(totalSteps):
  if stepCounter%1==0 and pId==0: printProgressTime( stepCounter, totalSteps,  computeTime + transferTime + saveTime )
  if stepCounter%4==0:
    #Save positions
    start.record()
    posHD.create_dataset( '{0:03}'.format(stepCounter/4), data=posSend_h )
    end.record(), end.synchronize()
    saveTime += start.time_till(end)*1e-3
  timeStep()
  
 
if pId == 0:
  end.record(), end.synchronize()
  totalTimeReal = startT.time_till(end)*1e-3
  totalTime = computeTime + transferTime + saveTime
  h, m, s = timeSplit( totalTime )
  print '\n\nTotal time: {0}:{1:02}:{2:02} '.format( h, m, s )
  print 'Compute  time: {0} secs   {1:2.2f}%  '.format( int(computeTime), 100*computeTime/totalTime )
  print 'Transfer time: {0} secs   {1:2.2f}%  '.format( int(transferTime), 100*transferTime/totalTime )
  print 'Save time: {0} secs   {1:2.2f}%  '.format( int(saveTime), 100*saveTime/totalTime )
  print 'Dead time: {0:.3f} secs'.format( totalTimeReal-totalTime )
  print'\n'

######################################################################
#Clean and Finalize
MPIcomm.Barrier()
dFile.close()
#Terminate CUDA
cudaCtx.pop()
cudaCtx.detach() #delete it
#Terminate MPI
MPIcomm.Barrier()
MPI.Finalize()
print "##########################################################END-{0}".format(pId)









