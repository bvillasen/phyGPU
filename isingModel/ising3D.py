# 3D Ising model simulation 
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAising3D.cu, volumeRender.py, CUDAvolumeRender.cu, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/volumeRender
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
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )

import volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray

nPoints = 256
useDevice = None
for option in sys.argv:
  #if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

temp = 2.
beta = np.float32( 1./temp)

plotVar = 1
upVal = 0.7
downVal = 0.4

#Initialize openGL
volumeRender.nWidth = nWidth
volumeRender.nHeight = nHeight
volumeRender.nDepth = nDepth
volumeRender.windowTitle = "Ising3D   spins={0}x{1}x{2}    T={3:.1f}".format(nHeight, nWidth, nDepth, float(temp))
volumeRender.initGL()

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
grid3D_ising = (gridx//2, gridy, gridz)


#initialize pyCUDA context 

cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "\nCompiling CUDA code"
cudaCodeString_raw = open("CUDAising3D.cu", "r").read()
cudaCodeString = cudaCodeString_raw # % { "BLOCK_WIDTH":block2D[0], "BLOCK_HEIGHT":block2D[1], "BLOCK_DEPTH":block2D[2], }
cudaCode = SourceModule(cudaCodeString)
tex_spins = cudaCode.get_texref('tex_spinsIn')
surf_spins = cudaCode.get_surfref('surf_spinsOut')
isingKernel = cudaCode.get_function('ising_kernel')
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
changeIntToFloat = ElementwiseKernel(arguments="float a, float b, int *input, float *output",
			      operation = "output[i] = a*input[i] + b;",
			      name = "intToFloat_kernel")
########################################################################
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################
def sendToScreen( plotData ):
  floatToUchar( plotDataFloat_d, plotData_d )
  copyToScreenArray()
########################################################################
def swipe():
  randomNumbers_d = curandom.rand((nData))
  stepNumber = np.int32(0)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  surf_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), beta, 
	       spinsOut_d, randomNumbers_d, 
	       plotDataFloat_d, np.float32(upVal), np.float32(downVal), grid=grid3D_ising, block=block3D )
  #copy3D_dtod() 

  stepNumber = np.int32(1)
  #saveEnergy = np.int32(0)
  tex_spins.set_array( spinsInArray_d )
  surf_spins.set_array( spinsInArray_d )
  isingKernel( stepNumber, np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), beta,
	       spinsOut_d, randomNumbers_d, 
	       plotDataFloat_d, np.float32(upVal), np.float32(downVal), grid=grid3D_ising, block=block3D )
  #copy3D_dtod()
########################################################################
def stepFunction():
  sendToScreen( spinsOut_d )
  swipe() 
########################################################################
def changePlotting():
  global upVal, downVal
  if plotVar == 1: upVal, downVal = 0.7, 0.4
  if plotVar == 2: upVal, downVal = 0.7, 100.
  if plotVar == 3: upVal, downVal = 0, 0.4
########################################################################
def specialKeyboardFunc( key, x, y ):
  global temp, beta, plotVar
  if key== volumeRender.GLUT_KEY_UP:
    temp += 0.1
  if key== volumeRender.GLUT_KEY_DOWN:
    if temp > 0.1: temp -= 0.1
  if key== volumeRender.GLUT_KEY_RIGHT:
    plotVar += 1
    if plotVar == 4: plotVar = 1
  if key== volumeRender.GLUT_KEY_LEFT:
    plotVar -= 1
    if plotVar == 0: plotVar = 3    
  beta = np.float32(1./temp)
  changePlotting()
  volumeRender.windowTitle = "Ising3D   spins={0}x{1}x{2}    T={3:.1f}".format(nHeight, nWidth, nDepth, float(temp))
########################################################################
########################################################################
#Initialize all gpu data
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )  
#Set initial random distribution
spins_h = (2*np.random.random_integers(0,1,[nDepth, nHeight, nWidth ]) - 1 ).astype(np.int32)
#spins_h = np.ones([nDepth, nHeight, nWidth ]).astype(np.int32)
spinsOut_d = gpuarray.to_gpu( spins_h )
randomNumbers_d = curandom.rand((nData))
#For texture version
spinsInArray_d, copy3D_dtod = gpuArray3DtocudaArray( spinsOut_d, allowSurfaceBind=True )
#For shared version
#memory for plotting
plotDataFloat_d = gpuarray.to_gpu(np.zeros_like(spins_h))
plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6) 


#configure volumeRender functions 
volumeRender.stepFunc = stepFunction
volumeRender.specialKeys = specialKeyboardFunc


#stepFunction()

#run volumeRender animation
volumeRender.animate()
