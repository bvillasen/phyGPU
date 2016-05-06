import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
#import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import h5py as h5
import matplotlib.pyplot as plt

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
dataDir = "/home/bruno/Desktop/data/qTurbulence/"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )


from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo
from tools import ensureDirectory, printProgressTime

cudaP = "double"
nPoints = 128
useDevice = None
usingAnimation = True
showKernelMemInfo = False
usingGravity = False

for option in sys.argv:
  if option == "grav": usingGravity = True
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1])
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) }
cudaPre, cudaPreComplex = precision[cudaP]


#set simulation volume dimentions
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

Lx = 1.
Ly = 1.
Lz = 1.
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
xPoints = X[0,0,:]
yPoints = Y[0,:,0]
zPoints = Z[0,0,:]
R = np.sqrt( X*X + Y*Y + Z*Z )
sphereR = 0.25
sphereOffCenter = 0.05
sphere = np.sqrt( (X)*(X) + Y*Y + Z*Z ) < 0.2
sphere_left  = ( np.sqrt( (X+sphereOffCenter)*(X+sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
sphere_right = ( np.sqrt( (X-sphereOffCenter)*(X-sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
spheres = sphere_right + sphere_left

gamma = 7./5.
c0 = 0.5

#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
pi4 = cudaPre( 4*np.pi )

#Initialize openGL
if usingAnimation:
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Hydro 3D  nPoints={0}".format(nPoints)
  volumeRender.initGL()

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 32,4,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]
grid3D_poisson = (gridx//2, gridy, gridz)
nPointsBlock = block3D[0]*block3D[1]*block3D[2]
nBlocksGrid = gridx * gridy * gridz
block2D = ( 16, 16, 1 )
grid2D = ( nWidth/block2D[0],  nHeight/block2D[1], 1 )

print "\nCompiling CUDA code"
cudaCodeFile = open("cuda_hydro_3D.cu","r")
cudaCodeString = cudaCodeFile.read().replace( "cudaP", cudaP )
cudaCodeString = cudaCodeString.replace( "THREADS_PER_BLOCK", str(nPointsBlock) )
  #"B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2],
  #'blockDim.x': block3D[0], 'blockDim.y': block3D[1], 'blockDim.z': block3D[2],
  #'gridDim.x': grid3D[0], 'gridDim.y': grid3D[1], 'gridDim.z': grid3D[2] }
cudaCode = SourceModule(cudaCodeString)
#setFlux_kernel = cudaCode.get_function('setFlux')
setInterFlux_hll_kernel = cudaCode.get_function('setInterFlux_hll')
getInterFlux_hll_kernel = cudaCode.get_function('getInterFlux_hll')
iterPoissonStep_kernel = cudaCode.get_function('iterPoissonStep')
getGravityForce_kernel = cudaCode.get_function('getGravityForce')
getBounderyPotential_kernel = cudaCode.get_function('getBounderyPotential')
reduceDensity_kernel = cudaCode.get_function('reduceDensity' )
tex_1 = cudaCode.get_texref("tex_1")
tex_2 = cudaCode.get_texref("tex_2")
tex_3 = cudaCode.get_texref("tex_3")
tex_4 = cudaCode.get_texref("tex_4")
tex_5 = cudaCode.get_texref("tex_5")
surf_1 = cudaCode.get_surfref("surf_1")
surf_2 = cudaCode.get_surfref("surf_2")
surf_3 = cudaCode.get_surfref("surf_3")
surf_4 = cudaCode.get_surfref("surf_4")
surf_5 = cudaCode.get_surfref("surf_5")
########################################################################
convertToUCHAR = ElementwiseKernel(arguments="cudaP normaliztion, cudaP *values, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*( values[i]*normaliztion -1 ) );",
			      name = "sendModuloToUCHAR_kernel")
########################################################################
getTimeMin_kernel = ReductionKernel( np.dtype( cudaPre ),
			    neutral = "1e6",
			    arguments=" float delta, cudaP* cnsv_rho, cudaP* cnsv_vel, float* soundVel".replace("cudaP", cudaP),
			    map_expr = " delta / ( abs( cnsv_vel[i]/ cnsv_rho[i] ) +  soundVel[i]   )    ",
			    reduce_expr = "min(a,b)",
			    name = "getTimeMin_kernel")
###################################################
def timeStepHydro():
  for coord in [ 1, 2, 3]:
    #Bind textures to read conserved
    tex_1.set_array( cnsv1_array )
    tex_2.set_array( cnsv2_array )
    tex_3.set_array( cnsv3_array )
    tex_4.set_array( cnsv4_array )
    tex_5.set_array( cnsv5_array )
    #Bind surfaces to write inter-cell fluxes
    surf_1.set_array( flx1_array )
    surf_2.set_array( flx2_array )
    surf_3.set_array( flx3_array )
    surf_4.set_array( flx4_array )
    surf_5.set_array( flx5_array )
    setInterFlux_hll_kernel( np.int32( coord ), cudaPre( gamma ), cudaPre(dx), cudaPre(dy), cudaPre(dz), cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d, times_d,  grid=grid3D, block=block3D )
    if coord == 1:
      dt = c0 * gpuarray.min( times_d ).get()
      print dt
    #Bind textures to read inter-cell fluxes
    tex_1.set_array( flx1_array )
    tex_2.set_array( flx2_array )
    tex_3.set_array( flx3_array )
    tex_4.set_array( flx4_array )
    tex_5.set_array( flx5_array )
    getInterFlux_hll_kernel( np.int32( coord ), cudaPre( dt ), cudaPre( gamma ), cudaPre(dx), cudaPre(dy), cudaPre(dz),
                          cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d,
                          gForceX_d, gForceY_d, gForceZ_d, gravWork_d, grid=grid3D, block=block3D )
    copy3D_cnsv1()
    copy3D_cnsv2()
    copy3D_cnsv3()
    copy3D_cnsv4()
    copy3D_cnsv5()
########################################################################
def solvePoisson( show=False ):
  maxIter = 1000
  for n in range(maxIter):
    converged.set( one_Array )
    tex_1.set_array( phi_array_1 )
    surf_1.set_array( phi_array_2 )
    iterPoissonStep_kernel( converged, np.int32( 0 ), np.int32( nWidth ), cudaPre( omega ), pi4,
			  cudaPre( dx ), cudaPre(dy), cudaPre(dz),
			  cnsv1_d, phi_d, phiWall_l_d, grid=grid3D_poisson, block=block3D )
    tex_1.set_array( phi_array_2 )
    surf_1.set_array( phi_array_1 )
    iterPoissonStep_kernel( converged, np.int32( 1 ), np.int32( nWidth ), cudaPre( omega ), pi4,
			  cudaPre( dx ), cudaPre(dy), cudaPre(dz),
			  cnsv1_d, phi_d, phiWall_l_d, grid=grid3D_poisson, block=block3D )
    copy3D_phi_1()
    copy3D_phi_2()
    if converged.get()[0] == 1:
      if show: print 'Poisson converged: ', n+1
      return
  if show: print 'Poisson converged: ', maxIter
########################################################################
def getGravForce( showConverIter=False):
  solvePoisson( show=showConverIter )
  tex_1.set_array( phi_array_1 )
  getGravityForce_kernel( np.int32( nWidth ), np.int32( nHeight ), np.int32( nDepth ),
		  dx, dy, dz, gForceX_d, gForceY_d, gForceZ_d,
		  cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, gravWork_d, phiWall_l_d, grid=grid3D, block=block3D )
########################################################################



def stepFuntion():
  maxVal = ( gpuarray.max( cnsv1_d ) ).get()
  convertToUCHAR( cudaPre( 0.95/maxVal ), cnsv1_d, plotData_d)
  copyToScreenArray()

  timeStepHydro()
  if usingGravity: getGravForce()

########################################################################
if showKernelMemInfo:
  #kernelMemoryInfo( setFlux_kernel, 'setFlux_kernel')
  #print ""
  kernelMemoryInfo( setInterFlux_hll_kernel, 'setInterFlux_hll_kernel')
  print ""
  kernelMemoryInfo( getInterFlux_hll_kernel, 'getInterFlux_hll_kernel')
  print ""
  kernelMemoryInfo( iterPoissonStep_kernel, 'iterPoissonStep_kernel')
  print ""
  kernelMemoryInfo( getBounderyPotential_kernel, 'getBounderyPotential_kernel')
  print ""
  kernelMemoryInfo( reduceDensity_kernel, 'reduceDensity_kernel')
  print ""
########################################################################
########################################################################
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )
rho = np.zeros( X.shape, dtype=cudaPre )  #density
vx  = np.zeros( X.shape, dtype=cudaPre )
vy  = np.zeros( X.shape, dtype=cudaPre )
vz  = np.zeros( X.shape, dtype=cudaPre )
p   = np.zeros( X.shape, dtype=cudaPre )  #pressure
#####################################################
#Initialize a centerd sphere
overDensity = sphere
rho[ overDensity ] = 1.
rho[ np.negative(overDensity) ] = 0.6
overPresure = sphere
p[ overPresure ] = 10
p[ np.negative(overPresure) ] = 1
v2 = vx*vx + vy*vy + vz*vz
#####################################################
#Initialize conserved values
cnsv1_h = rho
cnsv2_h = rho * vx
cnsv3_h = rho * vy
cnsv4_h = rho * vz
cnsv5_h = rho*v2/2. + p/(gamma-1)
#phi_h   = np.ones_like( rho )
#phi_h   = np.random.rand( nWidth, nHeight, nDepth ).astype( cudaPre )
#phi_h = -1./R.astype( cudaPre )
#phi_h   = np.zeros_like( rho )
phi_h   =  rho
gForce_h   = np.zeros_like( rho )

#####################################################
#Initialize device global data
cnsv1_d = gpuarray.to_gpu( cnsv1_h )
cnsv2_d = gpuarray.to_gpu( cnsv2_h )
cnsv3_d = gpuarray.to_gpu( cnsv3_h )
cnsv4_d = gpuarray.to_gpu( cnsv4_h )
cnsv5_d = gpuarray.to_gpu( cnsv5_h )
times_d = gpuarray.to_gpu( np.zeros( X.shape, dtype=np.float32 ) )
#For Gravitational potential
phi_d   = gpuarray.to_gpu( phi_h )
omega = 2. / ( 1 + np.pi / nWidth  )
one_Array = np.array([ 1 ]).astype( np.int32 )
converged = gpuarray.to_gpu( one_Array )
gForceX_d = gpuarray.to_gpu( gForce_h )
gForceY_d = gpuarray.to_gpu( gForce_h )
gForceZ_d = gpuarray.to_gpu( gForce_h )
gravWork_d = gpuarray.to_gpu( gForce_h )
phiWall_l_d = gpuarray.to_gpu( np.zeros( (nHeight, nDepth), dtype=np.float32 ) )
phiWall_r_d = gpuarray.to_gpu( np.zeros( (nHeight, nDepth), dtype=np.float32 ) )
rhoReduced_d = gpuarray.to_gpu( np.zeros( nBlocksGrid, dtype=np.float32 ) )
blockX_d = gpuarray.to_gpu( np.zeros( nBlocksGrid, dtype=np.float32 ) )
blockY_d = gpuarray.to_gpu( np.zeros( nBlocksGrid, dtype=np.float32 ) )
blockZ_d = gpuarray.to_gpu( np.zeros( nBlocksGrid, dtype=np.float32 ) )


#Texture and surface arrays
cnsv1_array, copy3D_cnsv1 = gpuArray3DtocudaArray( cnsv1_d, allowSurfaceBind=True, precision=cudaP )
cnsv2_array, copy3D_cnsv2 = gpuArray3DtocudaArray( cnsv2_d, allowSurfaceBind=True, precision=cudaP )
cnsv3_array, copy3D_cnsv3 = gpuArray3DtocudaArray( cnsv3_d, allowSurfaceBind=True, precision=cudaP )
cnsv4_array, copy3D_cnsv4 = gpuArray3DtocudaArray( cnsv4_d, allowSurfaceBind=True, precision=cudaP )
cnsv5_array, copy3D_cnsv5 = gpuArray3DtocudaArray( cnsv5_d, allowSurfaceBind=True, precision=cudaP )

flx1_array, copy3D_flx1_1 = gpuArray3DtocudaArray( cnsv1_d, allowSurfaceBind=True, precision=cudaP )
flx2_array, copy3D_flx2_1 = gpuArray3DtocudaArray( cnsv2_d, allowSurfaceBind=True, precision=cudaP )
flx3_array, copy3D_flx3_1 = gpuArray3DtocudaArray( cnsv3_d, allowSurfaceBind=True, precision=cudaP )
flx4_array, copy3D_flx4_1 = gpuArray3DtocudaArray( cnsv4_d, allowSurfaceBind=True, precision=cudaP )
flx5_array, copy3D_flx5_1 = gpuArray3DtocudaArray( cnsv5_d, allowSurfaceBind=True, precision=cudaP )
#Arrays for gravitational potential; checkboard iteration 2 arrays
phi_array_1, copy3D_phi_1 = gpuArray3DtocudaArray( phi_d, allowSurfaceBind=True, precision=cudaP )
phi_array_2, copy3D_phi_2 = gpuArray3DtocudaArray( phi_d, allowSurfaceBind=True, precision=cudaP )

if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6)

if usingGravity:
  print 'Getting initial Gravity Force...'
  start, end = cuda.Event(), cuda.Event()
  start.record() # start timing
  getGravForce( showConverIter=True )
  end.record(), end.synchronize()
  secs = start.time_till( end )*1e-3
  print 'Time: {0:0.4f}\n'.format( secs )




#plt.figure( 1 )
#phi =  phi_d.get()
#plt.imshow( phi[nDepth/2,:,:], extent=[xMin, xMax, yMin, yMax] )
#plt.colorbar()
##plt.show()

#plt.figure( 2 )
#forceX =  gForceX_d.get()
#forceY =  gForceY_d.get()
#forceZ =  gForceZ_d.get()
#force = np.sqrt( forceX*forceX + forceY*forceY + forceZ*forceZ )
#plt.imshow( force[nDepth/2,:,:], extent=[xMin, xMax, yMin, yMax] )
#plt.colorbar()

#plt.figure( 3 )
#plt.plot( xPoints, phi[nDepth/2,nHeight/2, :] )

#plt.figure( 4 )
#plt.plot( xPoints, forceX[nDepth/2,nHeight/2, :] )





#for i in range(500):
  #timeStepHydro()
  #if usingGravity: getGravForce()


#getGravForce()

#plt.figure( 5 )
#phi =  phi_d.get()
#plt.imshow( phi[nDepth/2,:,:], extent=[xMin, xMax, yMin, yMax] )
#plt.colorbar()
##plt.show()

#plt.figure( 6 )
#forceX =  gForceX_d.get()
#forceY =  gForceY_d.get()
#forceZ =  gForceZ_d.get()
#force = np.sqrt( forceX*forceX + forceY*forceY + forceZ*forceZ )
#plt.imshow( force[nDepth/2,:,:], extent=[xMin, xMax, yMin, yMax] )
#plt.colorbar()

#plt.figure( 7 )
#plt.plot( xPoints, phi[nDepth/2,nHeight/2, :] )

#plt.figure( 8 )
#plt.plot( xPoints, forceX[nDepth/2,nHeight/2, :] )


#plt.show()

























#from mpl_toolkits.mplot3d import Axes3D



#x = blockX_d.get()
#y = blockY_d.get()
#z = blockZ_d.get()


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z)
#plt.show()


#configure volumeRender functions
if usingAnimation:
  #volumeRender.viewTranslation[2] = -2
  volumeRender.transferScale = np.float32( 2.8 )
  #volumeRender.keyboard = keyboard
  #volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()
