# 2D Ising model simulation 
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAising2D.cu, animation2D.py, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/animation2D
#                               https://github.com/bvillasen/tools
import sys, time, os
import numpy as np
#import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
animation2DDirectory = parentDirectory + "/animation2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )

import animation2D
from cudaTools import setCudaDevice, getFreeMemory, gpuArray2DtocudaArray

nPoints = 1024*4
useDevice = None
for option in sys.argv:
  #if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints 
nData = nWidth*nHeight

temp = 1.
beta = np.float32( 1./temp)

#Initialize openGL
animation2D.nWidth = nWidth
animation2D.nHeight = nHeight
animation2D.windowTitle = "Ising Model 2D  spins={0}x{1}   T={2:.1f}".format(nHeight, nWidth, float(temp))
animation2D.initGL()

#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 16   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )  
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
block2D = (block_size_x, block_size_y, 1)
grid2D = (gridx, gridy, 1)
grid2D_ising = ( gridx//2, gridy, 1 )  #special grid to avoid neighbor conflicts 

#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "\nCompiling CUDA code"
cudaCodeString_raw = open("CUDAising2D.cu", "r").read()
cudaCodeString = cudaCodeString_raw # % { "BLOCK_WIDTH":block2D[0], "BLOCK_HEIGHT":block2D[1], "BLOCK_DEPTH":block2D[2], }
cudaCode = SourceModule(cudaCodeString)
tex_spins = cudaCode.get_texref('tex_spinsIn')
isingKernel = cudaCode.get_function('ising_kernel')
########################################################################
def sendToScreen( plotData ):
  floatToUchar( plotData, plotData_d )
  copyToScreenArray()
########################################################################
def swipe():
  randomNumbers_d = curandom.rand((nData))
  stepNumber = np.int32(0)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), beta, 
	       spinsOut_d, randomNumbers_d, grid=grid2D_ising, block=block2D )
  copy2D_dtod(aligned=True) 

  stepNumber = np.int32(1)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), beta,
	       spinsOut_d, randomNumbers_d, grid=grid2D_ising, block=block2D )
  copy2D_dtod(aligned=True)
########################################################################
def stepFunction():
  swipe()
########################################################################
def specialKeyboardFunc( key, x, y ):
  global temp, beta
  if key== animation2D.GLUT_KEY_UP:
    temp += 0.1
  if key== animation2D.GLUT_KEY_DOWN:
    if temp > 0.1: temp -= 0.1
  beta = np.float32(1./temp)
  animation2D.windowTitle = "Ising Model 2D  spins={0}x{1}   T={2:.1f}".format(nHeight, nWidth, float(temp))
  
########################################################################
########################################################################
#Initialize all gpu data
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )  
#Set initial random distribution
spins_h = (2*np.random.random_integers(0,1,[nHeight, nWidth]) - 1 ).astype(np.int32)
spinsOut_d = gpuarray.to_gpu( spins_h )
randomNumbers_d = curandom.rand((nData))
#For texture version
spinsInArray_d, copy2D_dtod = gpuArray2DtocudaArray( spinsOut_d )
#For shared version
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6) 
########################################################################
########################################################################


#configure animation2D functions and plotData
animation2D.stepFunc = stepFunction
animation2D.specialKeys = specialKeyboardFunc
animation2D.plotData_d = spinsOut_d
animation2D.maxVar = np.float32(2)
animation2D.minVar = np.float32(-20)

#stepFunction()

#run animation
animation2D.animate()


