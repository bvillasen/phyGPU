import sys, time, os
import numpy as np
#import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom
from pycuda.reduction import ReductionKernel
import h5py as h5

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
dataDir = "/home/bruno/Desktop/data/qTurbulence/"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )


from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo
from tools import ensureDirectory, printProgressTime


cudaP = "float"
nPoints = 128
useDevice = None
usingAnimation = False
realFFT = False
realTEXTURE = True
showKernelMemInfo = False
plotVelocity = False
for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1])
  if option == "fft": realFFT = True
  if option == "tex": realTEXTURE = True
  if option == "vel": plotVelocity = True
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) }
cudaPre, cudaPreComplex = precision[cudaP]

#set simulation volume dimentions
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

#Simulation Parameters
timeRelax = 5
dtImag = 0.005
#dtImag = 0.0001
dtReal = 0.005

Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]

omega = 0.5
alpha = 1000.0
gammaX = 1
gammaY = 1
gammaZ = 1
x0 = cudaPre( 0. )
y0 = cudaPre( 0. )
neighbors = 1

nIteratiosPerFrame_real = 50
if realTEXTURE: nIteratiosPerFrame_real = 50
nIteratiosPerFrame_imag = 20

applyTransition = False
realDynamics = False
plottingActive = True
plotVar = 0

#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
dtImag, dtReal = cudaPre(dtImag), cudaPre(dtReal)
omega, gammaX, gammaY, gammaZ = cudaPre(omega), cudaPre(gammaX), cudaPre(gammaY), cudaPre(gammaZ),
#Initialize openGL
if usingAnimation:
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Quantum Turbulence  nPoints={0}".format(nPoints)
  volumeRender.nTextures = 1 + 1*plotVelocity
  #volumeRender.viewXmin, volumeRender.viewXmax = -1., 1.
  #volumeRender.viewYmin, volumeRender.viewYmax = -1., 1.
  #volumeRender.viewZmin, volumeRender.viewZmax = -1., 1.
  volumeRender.initGL()

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]

print "\nCompiling CUDA code"
cudaCodeFile = open("CUDA_quantum.cu","r")
cudaCodeString_raw = cudaCodeFile.read().replace( "cudaP", cudaP )
cudaCodeString = cudaCodeString_raw % {
  "THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2],
  "B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2],
  'blockDim.x': block3D[0], 'blockDim.y': block3D[1], 'blockDim.z': block3D[2],
  'gridDim.x': grid3D[0], 'gridDim.y': grid3D[1], 'gridDim.z': grid3D[2] }
cudaCode = SourceModule(cudaCodeString)
getAlphas = cudaCode.get_function( "getAlphas_kernel" )
getFFTderivatives = cudaCode.get_function( "getFFTderivatives_kernel" ) #V_FFT
getPartialsXY = cudaCode.get_function( "getPartialsXY_kernel" )
setBoundryConditionsKernel = cudaCode.get_function( 'setBoundryConditions_kernel' )
implicitStep1 = cudaCode.get_function( "implicitStep1_kernel" )
implicitStep2 = cudaCode.get_function( "implicitStep2_kernel" )
findActivityKernel = cudaCode.get_function( "findActivity_kernel" )
getActivityKernel = cudaCode.get_function( "getActivity_kernel" )
getVelocityKernel = cudaCode.get_function( "getVelocity_kernel" )
eulerStepKernel = cudaCode.get_function( "eulerStep_kernel" )
eulerStep_FFTKernel = cudaCode.get_function( "eulerStep_fft_kernel" )  ##V_FFT
#TEXTURE version
eulerStep_texKernel = cudaCode.get_function( "eulerStep_texture_kernel" )
getVelocity_texKernel = cudaCode.get_function( "getVelocity_tex_kernel" )
tex_psiReal = cudaCode.get_texref("tex_psiReal")
tex_psiImag = cudaCode.get_texref("tex_psiImag")
surf_psiReal = cudaCode.get_surfref("surf_psiReal")
surf_psiImag = cudaCode.get_surfref("surf_psiImag")
if showKernelMemInfo:
  kernelMemoryInfo(eulerStepKernel, 'eulerStepKernel')
  print ""
  kernelMemoryInfo(eulerStep_texKernel, 'eulerStepKernel_texture')
  print ""

########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
########################################################################
multiplyByScalarComplex = ElementwiseKernel(arguments="cudaP a, pycuda::complex<cudaP> *psi".replace("cudaP", cudaP),
				operation = "psi[i] = a*psi[i] ",
				name = "multiplyByScalarComplex_kernel",
				preamble="#include <pycuda-complex.hpp>")
########################################################################
getModulo = ElementwiseKernel(arguments="pycuda::complex<cudaP> *psi, cudaP *psiMod".replace("cudaP", cudaP),
			      operation = "cudaP mod = abs(psi[i]);\
					    psiMod[i] = mod*mod;".replace("cudaP", cudaP),
			      name = "getModulo_kernel",
			      preamble="#include <pycuda-complex.hpp>")
########################################################################
sendModuloToUCHAR = ElementwiseKernel(arguments="cudaP *psiMod, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*(psiMod[i]-1));",
			      name = "sendModuloToUCHAR_kernel")
########################################################################
getNorm = ReductionKernel( np.dtype(cudaPre),
			    neutral = "0",
			    arguments=" cudaP dx, cudaP dy, cudaP dz, pycuda::complex<cudaP> * psi ".replace("cudaP", cudaP),
			    map_expr = "( conj(psi[i])* psi[i] )._M_re*dx*dy*dz",
			    reduce_expr = "a+b",
			    name = "getNorm_kernel",
			    preamble="#include <pycuda-complex.hpp>")
########################################################################
def gaussian3D(x, y, z, gammaX=1, gammaY=1, gammaZ=1, random=False):
  values =  np.exp( -gammaX*x*x - gammaY*y*y - gammaZ*z*z ).astype( cudaPre )
  if random:
    values += ( 100*np.random.random(values.shape) - 50 ) * values
  return values
########################################################################
def normalize( dx, dy, dz, complexArray ):
  factor = cudaPre( 1./(np.sqrt(getNorm(  dx, dy, dz, complexArray ).get())) )  #OPTIMIZATION
  multiplyByScalarComplex( factor, complexArray )
########################################################################
def implicit_iteration( ):
  global alpha
  #Make FFT
  fftPlan.execute( psi_d, psiFFT_d )
  #get Derivatives
  getPartialsXY( Lx, Ly, psiFFT_d, partialX_d, fftKx_d, partialY_d, fftKy_d, block=block3D, grid=grid3D)
  fftPlan.execute( partialX_d, inverse=True )
  fftPlan.execute( partialY_d, inverse=True )
  implicitStep1( xMin, yMin, zMin, dx, dy, dz, alpha,  omega,  gammaX,  gammaY,  gammaZ,
		  partialX_d, partialY_d, psi_d, G_d, x0, y0, grid=grid3D, block=block3D)
  fftPlan.execute( G_d )
  implicitStep2( dtImag, fftKx_d , fftKy_d, fftKz_d, alpha, psiFFT_d, G_d, block=block3D, grid=grid3D)
  fftPlan.execute( psiFFT_d, psi_d, inverse=True)
  #setBoundryConditionsKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), psi_d, block=block3D, grid=grid3D)
  normalize(dx, dy, dz, psi_d)
  #GetAlphas
  getAlphas( dx, dy, dz, xMin, yMin, zMin, gammaX, gammaY, gammaZ, psi_d, alphas_d, block = block3D, grid=grid3D)
  alpha= cudaPre( ( 0.5*(gpuarray.max(alphas_d) + gpuarray.min(alphas_d)) ).get() )  #OPTIMIZACION
########################################################################
def imaginaryStep():
  [ implicit_iteration() for i in range(nIteratiosPerFrame_imag) ]
########################################################################
def rk4_iteration():
  cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
  findActivityKernel( cudaPre(0.00001), psi_d, activity_d, grid=grid3D, block=block3D )
  #Step 1
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 0.5 )
  eulerStepKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiK2_d, psiK1_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 2
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 0.5 )
  eulerStepKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiK1_d, psiK2_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 3
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 1. )
  eulerStepKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiK2_d, psiK1_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 4
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 1. )
  eulerStepKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiK1_d, psiK2_d, psiRunge_d, np.uint8(1), activity_d, grid=grid3D, block=block3D )
########################################################################
def rk4_texture_iteration():
  cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
  findActivityKernel( cudaPre(0.00001), psi_d, activity_d, grid=grid3D, block=block3D )
  #Step 1
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 0.5 )
  tex_psiReal.set_array( psiK2Real_array )
  tex_psiImag.set_array( psiK2Imag_array )
  surf_psiReal.set_array( psiK1Real_array )
  surf_psiImag.set_array( psiK1Imag_array )
  eulerStep_texKernel( slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 2
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 0.5 )
  tex_psiReal.set_array( psiK1Real_array )
  tex_psiImag.set_array( psiK1Imag_array )
  surf_psiReal.set_array( psiK2Real_array )
  surf_psiImag.set_array( psiK2Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 3
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 1. )
  tex_psiReal.set_array( psiK2Real_array )
  tex_psiImag.set_array( psiK2Imag_array )
  surf_psiReal.set_array( psiK1Real_array )
  surf_psiImag.set_array( psiK1Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiRunge_d, np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 4
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 1. )
  tex_psiReal.set_array( psiK1Real_array )
  tex_psiImag.set_array( psiK1Imag_array )
  surf_psiReal.set_array( psiK2Real_array )
  surf_psiImag.set_array( psiK2Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, omega,
		  psi_d, psiRunge_d, np.uint8(1), activity_d, grid=grid3D, block=block3D )
########################################################################
def rk4_FFT_iteration():
  cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
  findActivityKernel( cudaPre(0.00001), psi_d, activity_d, grid=grid3D, block=block3D )
  #Step 1
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 0.5 )
  fftPlan.execute( psiK2_d, psiFFT_d )
  getFFTderivatives( Lx, Ly, Lz, psiFFT_d, fftKx_d, fftKy_d, fftKz_d, partialX_d, partialY_d, laplacian_d, grid=grid3D, block=block3D )
  fftPlan.execute( partialX_d,  inverse=True )
  fftPlan.execute( partialY_d,  inverse=True )
  fftPlan.execute( laplacian_d, inverse=True )
  eulerStep_FFTKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, x0, y0, omega,
		  psi_d, psiK2_d, psiK1_d, psiRunge_d, laplacian_d, partialX_d, partialY_d,
		  np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 2
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 0.5 )
  fftPlan.execute( psiK1_d, psiFFT_d )
  getFFTderivatives( Lx, Ly, Lz, psiFFT_d, fftKx_d, fftKy_d, fftKz_d, partialX_d, partialY_d, laplacian_d, grid=grid3D, block=block3D )
  fftPlan.execute( partialX_d,  inverse=True )
  fftPlan.execute( partialY_d,  inverse=True )
  fftPlan.execute( laplacian_d, inverse=True )
  eulerStep_FFTKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, x0, y0, omega,
		  psi_d, psiK1_d, psiK2_d, psiRunge_d, laplacian_d, partialX_d, partialY_d,
		  np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 3
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 1. )
  fftPlan.execute( psiK2_d, psiFFT_d )
  getFFTderivatives( Lx, Ly, Lz, psiFFT_d, fftKx_d, fftKy_d, fftKz_d, partialX_d, partialY_d, laplacian_d, grid=grid3D, block=block3D )
  fftPlan.execute( partialX_d,  inverse=True )
  fftPlan.execute( partialY_d,  inverse=True )
  fftPlan.execute( laplacian_d, inverse=True )
  eulerStep_FFTKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, x0, y0, omega,
		  psi_d, psiK2_d, psiK1_d, psiRunge_d, laplacian_d, partialX_d, partialY_d,
		  np.uint8(0), activity_d, grid=grid3D, block=block3D )
  #Step 4
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 1. )
  fftPlan.execute( psiK1_d, psiFFT_d )
  getFFTderivatives( Lx, Ly, Lz, psiFFT_d, fftKx_d, fftKy_d, fftKz_d, partialX_d, partialY_d, laplacian_d, grid=grid3D, block=block3D )
  fftPlan.execute( partialX_d,  inverse=True )
  fftPlan.execute( partialY_d,  inverse=True )
  fftPlan.execute( laplacian_d, inverse=True )
  eulerStep_FFTKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ, x0, y0, omega,
		  psi_d, psiK1_d, psiK2_d, psiRunge_d, laplacian_d, partialX_d, partialY_d,
		  np.uint8(1), activity_d, grid=grid3D, block=block3D )
########################################################################
def realStep():
  if realTEXTURE: [rk4_texture_iteration() for i in range(nIteratiosPerFrame_real)]
  else: [rk4_iteration() for i in range( nIteratiosPerFrame_real )]
########################################################################
def stepFuntion():
  getModulo( psi_d, psiMod_d )
  maxVal = (gpuarray.max(psiMod_d)).get()
  multiplyByScalarReal( cudaPre(0.95/(maxVal)), psiMod_d )
  sendModuloToUCHAR( psiMod_d, plotData_d)
  copyToScreenArray()

  if volumeRender.nTextures == 2:
    if not realDynamics:
      cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
      findActivityKernel( cudaPre(0.001), psi_d, activity_d, grid=grid3D, block=block3D )
    if plotVar == 1: getActivityKernel( psiOther_d, activity_d, grid=grid3D, block=block3D )
    if plotVar == 0:
      if realTEXTURE:
	tex_psiReal.set_array( psiK2Real_array )
	tex_psiImag.set_array( psiK2Imag_array )
	getVelocity_texKernel( dx, dy, dz, psi_d, activity_d, psiOther_d, grid=grid3D, block=block3D )
      else: getVelocityKernel( np.int32(neighbors), dx, dy, dz, psi_d, activity_d, psiOther_d, grid=grid3D, block=block3D )
      maxVal = (gpuarray.max(psiOther_d)).get()
      if maxVal > 0: multiplyByScalarReal( cudaPre(1./maxVal), psiOther_d )
    sendModuloToUCHAR( psiOther_d, plotData_d_1)
    copyToScreenArray_1()
  if applyTransition: timeTransition()
  if realDynamics: realStep()
  else: imaginaryStep()

########################################################################
def timeTransition():
  global realDynamics, alpha, applyTransition
  realDynamics = not realDynamics
  applyTransition = False
  if realDynamics:
    cuda.memcpy_dtod(psiK2_d.ptr, psi_d.ptr, psi_d.nbytes)
    cuda.memcpy_dtod(psiRunge_d.ptr, psi_d.ptr, psi_d.nbytes)
    if realTEXTURE:
      copy3DpsiK1Real()
      copy3DpsiK1Imag()
      copy3DpsiK2Real()
      copy3DpsiK2Imag()
    print "Real Dynamics"
  else:
    #GetAlphas
    getAlphas( dx, dy, dz, xMin, yMin, zMin, gammaX, gammaY, gammaZ, psi_d, alphas_d, block = block3D, grid=grid3D)
    alpha= cudaPre( ( 0.5*(gpuarray.max(alphas_d) + gpuarray.min(alphas_d)) ).get() )  #OPTIMIZACION
    print "Imaginary Dynamics"
########################################################################
def loadState( fileName = "psi.hdf5" ):
  dataFile = h5.File( dataDir + fileName ,'r')
  dataAll = dataFile.get("psi")[...]
  print '\nLoading data... \n file: {0} \n '.format(fileName )
  return dataAll[...]
########################################################################
def saveState( fileName='psi.h5' ):
  ensureDirectory( dataDir )
  psi_h = psi_d.get()
  print "Saving Data"
  outputName = dataDir + fileName
  dataFile = h5.File( outputName  ,'w')
  dataFile.create_dataset( "psi", data=psi_d.get(), compression='lzf')
  dataFile.close()
  print "Data Saved: {0}\n".format( outputName )
#######################################################################

print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )
psi_h = np.zeros( X.shape, dtype=cudaPreComplex )
psi_h.real = gaussian3D ( X, Y, Z, gammaX, gammaY, gammaZ, random=True )
#####################################################
#Load Data
#psi_h = loadState().astype(cudaPreComplex)
#####################################################
print " Making FFT plan"
from pyfft.cuda import Plan
fftPlan = Plan((nDepth, nHeight, nWidth),  dtype=cudaPreComplex)
#from scikits.cuda.fft import fft, Plan
#fftPlan = Plan((nDepth, nHeight, nWidth),  in_dtype=cudaPreComplex, out_dtype=cudaPreComplex)
fftKx_h = np.zeros( nWidth, dtype=cudaPre )
fftKy_h = np.zeros( nHeight, dtype=cudaPre )
fftKz_h = np.zeros( nDepth, dtype=cudaPre )
for i in range(nWidth/2):
  fftKx_h[i] = i*2*np.pi/Lx
for i in range(nWidth/2, nWidth):
  fftKx_h[i] = (i-nWidth)*2*np.pi/Lx
for i in range(nHeight/2):
  fftKy_h[i] = i*2*np.pi/Ly
for i in range(nHeight/2, nHeight):
  fftKy_h[i] = (i-nHeight)*2*np.pi/Ly
for i in range(nDepth/2):
  fftKz_h[i] = i*2*np.pi/Lz
for i in range(nDepth/2, nDepth):
  fftKz_h[i] = (i-nDepth)*2*np.pi/Lz
psi_d = gpuarray.to_gpu(psi_h)
alphas_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
normalize( dx, dy, dz, psi_d )
getAlphas( dx, dy, dz, xMin, yMin, zMin, gammaX, gammaY, gammaZ, psi_d, alphas_d, block = block3D, grid=grid3D)
alpha=( 0.5*(gpuarray.max(alphas_d) + gpuarray.min(alphas_d)) ).get()
psiMod_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
psiFFT_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
partialX_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
partialY_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
G_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
fftKx_d = gpuarray.to_gpu( fftKx_h )         #OPTIMIZATION
fftKy_d = gpuarray.to_gpu( fftKy_h )
fftKz_d = gpuarray.to_gpu( fftKz_h )
activity_d = gpuarray.to_gpu( np.ones( nBlocks3D, dtype=np.uint8 ) )
psiOther_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
psiK1_d = gpuarray.to_gpu( psi_h )
psiK2_d = gpuarray.to_gpu( psi_h )
psiRunge_d = gpuarray.to_gpu( psi_h )
#For FFT version
laplacian_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
#TEXTURE version
k1tempReal = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
k1tempImag = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
k2tempReal = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
k2tempImag = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
psiK1Real_array, copy3DpsiK1Real = gpuArray3DtocudaArray( k1tempReal, allowSurfaceBind=True, precision=cudaP )
psiK1Imag_array, copy3DpsiK1Imag = gpuArray3DtocudaArray( k1tempImag, allowSurfaceBind=True, precision=cudaP )
psiK2Real_array, copy3DpsiK2Real = gpuArray3DtocudaArray( k2tempReal, allowSurfaceBind=True, precision=cudaP )
psiK2Imag_array, copy3DpsiK2Imag = gpuArray3DtocudaArray( k2tempImag, allowSurfaceBind=True, precision=cudaP )
#memory for plotting
if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
  if volumeRender.nTextures == 2:
    plotData_d_1 = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
    volumeRender.plotData_dArray_1, copyToScreenArray_1 = gpuArray3DtocudaArray( plotData_d_1 )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6)

def keyboard(*args):
  global plottingActive
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()
  if args[0] == '1':
    volumeRender.transferScale += np.float32(0.01)
    print "Image Transfer Scale: ", volumeRender.transferScale
  if args[0] == '2':
    volumeRender.transferScale -= np.float32(0.01)
    print "Image Transfer Scale: ",volumeRender.transferScale
  if args[0] == '4':
    volumeRender.brightness -= np.float32(0.1)
    print "Image Brightness : ",volumeRender.brightness
  if args[0] == '5':
    volumeRender.brightness += np.float32(0.1)
    print "Image Brightness : ",volumeRender.brightness
  if args[0] == '7':
    volumeRender.density -= np.float32(0.01)
    print "Image Density : ",volumeRender.density
  if args[0] == '8':
    volumeRender.density += np.float32(0.01)
    print "Image Density : ",volumeRender.density
  if args[0] == '3':
    volumeRender.transferOffset += np.float32(0.01)
    print "Image Offset : ", volumeRender.transferOffset
  if args[0] == '6':
    volumeRender.transferOffset -= np.float32(0.01)
    print "Image Offset : ", volumeRender.transferOffset
  if args[0] == 'a':
    plottingActive = not plottingActive
    if plottingActive: print "plottingActive"

def specialKeyboardFunc( key, x, y ):
  global plotVar, neighbors, plottingActive, applyTransition
  #global omega
  if key== volumeRender.GLUT_KEY_DOWN:
    plottingActive = not plottingActive
    if plottingActive: print "plottingActive"
    #omega -= cudaPre(0.01)
    #print omega
  if key== volumeRender.GLUT_KEY_RIGHT:
    plotVar += 1
    if plotVar == 2: plotVar = 0
  if key== volumeRender.GLUT_KEY_LEFT:
    applyTransition = True

######################################################################################
######################################################################################
if showKernelMemInfo:
  implicit_iteration()
  implicit_iteration()
  applyTransition = True
  timeTransition()
  if realTEXTURE:
    rk4_texture_iteration()
    rk4_texture_iteration()
  else:
    rk4_iteration()
    rk4_iteration()
  print "Precision: ", cudaP
  print "Timing Info saved in: cuda_profile_1.log \n\n"
  sys.exit()
######################################################################################
######################################################################################




#configure volumeRender functions
if usingAnimation:
  volumeRender.viewTranslation[2] = -2
  volumeRender.keyboard = keyboard
  volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()



print "nPoints: {0}x{1}x{2}".format(nWidth, nHeight, nWidth )
print "Starting Imaginary Dynamics: {0} timeUnits, dt: {1:1.2e} \n".format( timeRelax, dtImag )
simulationTime = -timeRelax
printCounter = 0
start, end = cuda.Event(), cuda.Event()
start.record()
while simulationTime < 0:
  if timeRelax+simulationTime >= printCounter*timeRelax/500.:
    end.record().synchronize()
    printProgressTime( timeRelax + simulationTime, timeRelax,  start.time_till(end)*1e-3 )
    printCounter += 1
  imaginaryStep()
  simulationTime += dtImag
print "\nEnd Imaginary Dynamics\n "
saveState( fileName='initial/converged/psiConverged.h5')

#applyTransition = True
#timeTransition()
##cuda.memset_d8(activity_d.ptr, 1, nBlocks3D )
#endTime = 1
#nIterations = int( endTime/dtReal )
#print "Starting Real Dynamics: {0} timeUnits ".format( endTime )
#[ rk4_iteration() for i in range(nIterations) ]
