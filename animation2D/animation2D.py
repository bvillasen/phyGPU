from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pyglew as glew

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
animationDirectory = parentDirectory + "/animation2D"
sys.path.extend( [animationDirectory] )


nWidth = 512
nHeight = 512

windowTitle = ''


gl_Tex = None
gl_PBO = None
cuda_POB = None
cuda_POB_ptr = None
colorMap_rgba_d = None
plot_rgba_d = None
plotData_d = None
background_h = None
background_d = None

get_rgbaKernel = None
copyKernel = None
block2D_GL = None
grid2D_GL = None 

showPoint = False
onePoint = ( 0. ,0. )

frames = 0
nCol = 236
maxVar = 1.001
minVar = 0.
nCol = np.int32( nCol )

usingDouble = False
cudaPrecision = "float"
usingGrayScale = False

frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

viewXmin_o, viewXmax_o = -nWidth//2, nWidth//2
viewYmin_o, viewYmax_o = -nHeight//2, nHeight//2


viewXmin, viewXmax = -nWidth//2, nWidth//2
viewYmin, viewYmax = -nHeight//2, nHeight//2
viewZmin, viewZmax = -200, 200

def initCUDA():
  global get_rgbaKernel, copyKernel, block2D_GL, grid2D_GL, maxVar, minVar, cudaPrecision
  block2D_GL = (16,16, 1)
  grid2D_GL = (nWidth/block2D_GL[0], nHeight/block2D_GL[1] )
  maxVar = np.float32(maxVar)
  minVar = np.float32(minVar)
  if usingDouble: 
    cudaPrecision = "double"
    maxVar = np.float64(maxVar)
    minVar = np.float64(minVar)
  cudaAnimCode = SourceModule('''
  #include <stdint.h>
  #include <cuda.h>

  __global__ void get_rgba_kernel (int ncol, %(cudaP)s minvar, %(cudaP)s maxvar, %(cudaP)s *plot_data, unsigned int *plot_rgba_data,
				  unsigned int *cmap_rgba_data, unsigned char *background){
  // CUDA kernel to fill plot_rgba_data array for plotting    
    int t_i = blockIdx.x*blockDim.x + threadIdx.x;
    int t_j = blockIdx.y*blockDim.y + threadIdx.y;
    int tid = t_i + t_j*blockDim.x*gridDim.x;

    float frac = (plot_data[tid]-minvar)/(maxvar-minvar);
    int icol = (int)(frac * ncol);
    plot_rgba_data[tid] = background[tid]*cmap_rgba_data[icol];
  }
  
  '''%{"cudaP": cudaPrecision})
  get_rgbaKernel = cudaAnimCode.get_function('get_rgba_kernel')
  print "CUDA 2D animation initialized"


def initData():  
  global plot_rgba_d, colorMap_rgba_d, plotData_d, background_d, background_h
  #print "Loading Color Map"
  colorMapFile = "/cmapGray.dat" if usingGrayScale else "/cmap.dat"
  colorMap = np.loadtxt(animationDirectory + colorMapFile)
  colorMap_rgba = []
  for i in range(colorMap.shape[0]):
    r, g, b = colorMap[i]
    colorMap_rgba.append( int(255)<<24 | int(b*255)<<16 | int(g*255)<<8 | int(r*255)<<0 )
  colorMap_rgba = np.array(colorMap_rgba)
  colorMap_rgba_h = np.array(colorMap_rgba).astype(np.uint32)
  colorMap_rgba_d = gpuarray.to_gpu( colorMap_rgba_h )
  plot_rgba_h = np.zeros(nWidth*nHeight).astype(np.uint32)
  plot_rgba_d = gpuarray.to_gpu( plot_rgba_h )
  plotData_h = np.random.rand(nWidth*nHeight).astype(np.float32) 
  if usingDouble: plotData_h = np.random.rand(nWidth*nHeight).astype(np.float64) 
  if background_h == None: background_h = np.ones( [nHeight, nWidth], dtype=np.uint8 )  
  if not background_d: background_d = gpuarray.to_gpu(background_h)
  if not plotData_d: plotData_d = gpuarray.to_gpu(plotData_h)

def get_rgba( ptr ):
  #global plotData_d, plot_rgba_d, colorMap_rgba_d
  get_rgbaKernel(nCol, minVar, maxVar, plotData_d, np.intp(ptr), colorMap_rgba_d, background_d, grid=grid2D_GL, block=block2D_GL )





def stepFunc():
  #print "Default step function"
  return 0


def displayFunc():
  global frames, timer
  global cuda_POB
 
  timer = time.time()
  frames += 1
  stepFunc() 
  cuda_POB_map = cuda_POB.map()
  cuda_POB_ptr, cuda_POB_size = cuda_POB_map.device_ptr_and_size()
  get_rgba( cuda_POB_ptr ) 
  cuda_POB_map.unmap()
  
  glClear(GL_COLOR_BUFFER_BIT) # Clear
  glBindTexture(GL_TEXTURE_2D, gl_Tex)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO)
  
  #NON-GL_INTEROPITAL
  #plot_rgba = plot_rgba_d.get()
  ##Fill the pixel buffer with the plot_rgba array
  #glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, plot_rgba.nbytes, plot_rgba, GL_STREAM_COPY)
  
  # Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,nWidth,nHeight,GL_RGBA,GL_UNSIGNED_BYTE,None)  
  
  #Render one quad to the screen and colour it using our texture
  #i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT)
  glBegin(GL_QUADS)
  glTexCoord2f (0.0, 0.0)
  glVertex3f (viewXmin, viewYmin, 0.0)
  glTexCoord2f (1.0, 0.0)
  glVertex3f (viewXmax, viewYmin, 0.0)
  glTexCoord2f (1.0, 1.0)
  glVertex3f (viewXmax, viewYmax, 0.0)
  glTexCoord2f (0.0, 1.0)
  glVertex3f (viewXmin, viewYmax, 0.0)
  glEnd()
  
  if showPoint:
    #Draw one point in top of image
    glPointSize(15)
    #glColor3f( 0.95, 0.207, 0.031 );
    glBegin(GL_POINTS)  
    #glColor3f( 0.95, 0.207, 0.031 );
    glVertex3f ( viewXmin_o+ onePoint[0]/2, viewYmin_o+ onePoint[1]/2, 1.)
    glEnd()
  
  timer = time.time()-timer
  computeFPS()
  glutSwapBuffers()

GL_initialized = False  
def initGL():
  global GL_initialized
  global viewXmin, viewXmax, viewYmin, viewYmax
  if GL_initialized: return
  glutInit()
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
  glutInitWindowSize(nWidth, nHeight)
  #glutInitWindowPosition(50, 50)
  glutCreateWindow("Window")
  #glew.glewInit()
  glClearColor(1.0, 0.0, 0.0, 0.0)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  viewXmin, viewXmax, viewYmin, viewYmax = viewXmin_o, viewXmax_o, viewYmin_o, viewYmax_o
  #glOrtho(0,nWidth,0.,nHeight, -200.0, 200.0)
  glOrtho( viewXmin, viewXmax, 
				viewYmin, viewYmax,
				viewZmin, viewZmax)
  GL_initialized = True
  print "\nOpenGL initialized"
  

  


def createPBO():
  global gl_Tex
  #Create texture which we use to display the result and bind to gl_Tex
  glEnable(GL_TEXTURE_2D)
  gl_Tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, gl_Tex)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nWidth, nHeight, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, None);
  #print "Texture Created"
# Create pixel buffer object and bind to gl_PBO. We store the data we want to
# plot in memory on the graphics card - in a "pixel buffer". We can then 
# copy this to the texture defined above and send it to the screen
  global gl_PBO, cuda_POB
  
  gl_PBO = glGenBuffers(1)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO)
  glBufferData(GL_PIXEL_UNPACK_BUFFER, nWidth*4*nHeight, None, GL_STREAM_COPY)
  cuda_POB = cuda_gl.RegisteredBuffer(long(gl_PBO))
  #cuda_POB_map = cuda_POB.map()
  #cuda_POB_ptr, cuda_POB_size = cuda_POB_map.device_ptr_and_size()
  #print "Buffer Created"

def computeFPS():
    global frameCount, fpsCount, fpsLimit, timer
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle(windowTitle + "      fps={0:0.2f}".format( float(ifps) ))
        fpsCount = 0

def startGL():
  glutDisplayFunc(displayFunc)
  glutReshapeFunc(resize)
  glutIdleFunc(displayFunc)
  glutKeyboardFunc( keyboard )
  glutSpecialFunc(specialKeys)
  glutMouseFunc(mouse)
  #glutIdleFunc( idleFunc )
  if backgroundType == 'point': glutMotionFunc(mouseMotion_point)
  if backgroundType == 'square': glutMotionFunc(mouseMotion_square)
  if backgroundType == 'move': glutMotionFunc(mouseMotion_move)
  #import pycuda.autoinit
  print "Starting GLUT main loop..."
  glutMainLoop()


def animate():
  global windowTitle
  if not GL_initialized: initGL()
  #import pycuda.gl.autoinit
  if windowTitle == '': windowTitle = "CUDA 2D animation"
  initCUDA()
  createPBO()
  initData()
  startGL()

nWidth_GL = nWidth
nHeight_GL = nHeight
def resize( w, h ):
  global nWidth_GL, nHeight_GL
  global zoomed
  nWidth_GL, nHeight_GL = w, h
  glViewport (0, 0, w, h)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity() 
  #glOrtho (0., nWidth, 0., nHeight, -200. ,200.)
  glOrtho( viewXmin, viewXmax, 
				viewYmin, viewYmax,
				viewZmin, viewZmax)
  glMatrixMode (GL_MODELVIEW)
  glLoadIdentity()
  zoomed = 0
  #print nWidth_GL, nHeight_GL
  
  



def keyboard(*args):
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.Context.pop()
    sys.exit() 

def specialKeys( key, x, y ):
  if key==GLUT_KEY_UP:
    print "UP-arrow pressed"
  if key==GLUT_KEY_DOWN:
    print "DOWN-arrow pressed"


iPosOld, jPosOld = None, None
backgroundFlag = 0
backgroundType = "point"
zoomed = 0
zoomFactor = 1.5
ox, oy = 0, 0

def mouse( button, state, x, y ):
  global iPosOld, jPosOld, backgroundFlag, backgroundType
  global jMin, jMax, iMin, iMax
  global viewXmin, viewXmax, viewYmin, viewYmax, zoomed
  global viewXmin_o, viewXmax_o, viewYmin_o, viewYmax_o, zoomed
  global ox, oy
  ox, oy = x, y
  if button == 3 and state == GLUT_DOWN:
    #print 'wheel up'
    viewXmin *= zoomFactor
    viewXmax *= zoomFactor
    viewYmin *= zoomFactor
    viewYmax *= zoomFactor
    #viewXmin_o *= zoomFactor
    #viewXmax_o *= zoomFactor
    #viewYmin_o *= zoomFactor
    #viewYmax_o *= zoomFactor
    zoomed += 1
    #resize(nWidth, nHeight)
    return
  if button == 4 and state == GLUT_DOWN:
    #print 'wheel up'
    if zoomed > 0:
      viewXmin /= zoomFactor
      viewXmax /= zoomFactor
      viewYmin /= zoomFactor
      viewYmax /= zoomFactor
      #viewXmin_o /= zoomFactor
      #viewXmax_o /= zoomFactor
      #viewYmin_o /= zoomFactor
      #viewYmax_o /= zoomFactor
      if viewXmax < viewXmax_o: viewXmax = viewXmax_o
      if viewXmin > viewXmin_o: viewXmin = viewXmin_o
      if viewYmax < viewYmax_o: viewYmax = viewYmax_o
      if viewYmin > viewYmin_o: viewYmin = viewYmin_o
      zoomed -= 1
      if zoomed == 0: viewXmax, viewXmin, viewYmax, viewYmin = viewXmax_o, viewXmin_o, viewYmax_o, viewYmin_o 
    #resize(nWidth, nHeight)
    return  
  xx, yy = float(x), float(y)
  if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN :
    jPosOld = int( xx/nWidth_GL*nWidth )
    iPosOld = int( (nHeight_GL - yy )/nHeight_GL*nHeight )
    backgroundFlag = 0
    #print iPosOld, jPosOld
  if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN :  
    jPosOld = int(xx/nWidth_GL*nWidth)
    iPosOld = int( ( nHeight_GL - yy )/nHeight_GL*nHeight)
    backgroundFlag = 1
    if backgroundType == "square":replotFunc()
  if backgroundType == "point":
    background_h[iPosOld, jPosOld] = backgroundFlag  
    background_d.set(background_h) 
    

def mouseMotion_move(x,y):
  global ox, oy
  global viewXmin, viewXmax, viewYmin, viewYmax
  dx, dy = x-ox, y-oy
  moveX = dx/8.*zoomed*(nWidth_GL/nWidth)
  moveY = -dy/8.*zoomed*(nHeight_GL/nHeight)
  if (viewXmin + moveX < viewXmin_o and viewXmax + moveX > viewXmax_o): 
    viewXmin += moveX
    viewXmax += moveX
  if (viewYmin + moveY < viewYmin_o and viewYmax + moveY > viewYmax_o): 
    viewYmin += moveY
    viewYmax += moveY
  ox, oy = x, y
   
def mouseMotion_point( x, y ):
  global jPosOld, iPosOld
  xx, yy = float(x), float(y)
  jPos = int( xx/nWidth_GL*nWidth )
  iPos = int( (nHeight_GL - yy )/nHeight_GL*nHeight )
  if (jPos < 0 or jPos >=nWidth) or iPos < 0 or iPos >=nHeight : return

  if jPos <= jPosOld:
    j1 = jPos
    j2 = jPosOld
    i1 = iPos
    i2 = iPosOld
  else:
    j1 = jPosOld
    j2 = jPos
    i1 = iPosOld
    i2 = iPos
  iLast = i1
  for j in range(j1,j2+1):
    if j1 != j2: 
      frac = float(j-j1)/float(j2-j1)
      iNext = int(frac*(i2-i1)) + i1
    else: iNext = i2
    if iNext >= iLast:
      background_h[iLast][j] = backgroundFlag
      for i in range( iLast, iNext+1 ):
	background_h[i][j] = backgroundFlag
    else:
      background_h[iLast][j] = backgroundFlag
      for i in range(iNext, iLast+1):
	background_h[i][j] = backgroundFlag
    iLast = iNext
  background_d.set(background_h)
  jPosOld, iPosOld = jPos, iPos

jMin = 10000
jMax = -1
iMin = 10000
iMax = -1   
def mouseMotion_square( x, y ):  
  global iPosOld, jPosOld
  global jMin, jMax, iMin, iMax
  x0, y0 = jPosOld, iPosOld
  xx = max( min( x, nWidth_GL ), 0.0 )
  yy = max( min( nHeight_GL - y, nHeight_GL ), 0.0 )
  
  xReal = float(xx)/nWidth_GL*nWidth
  yReal = float(yy)/nHeight_GL*nHeight
  
  jMin = min( x0, int(xReal))
  jMax = max( x0, int(xReal))
  iMin = min( y0, int(yReal))
  iMax = max( y0, int(yReal)) 
  mouseMaskFunc()
  #background_h[iMin:iMax, jMin:jMax] = 0
  #background_d.set(background_h)
  #print  jMin, jMax, iMin, iMax
  #print " Reploting: ( {0} , {1} , {2} , {3} )".format(jMin, jMax, iMin, iMax)
  
def replotFunc():
  print "No specified replot function"
  return 0

def mouseMaskFunc():
  print "No specified maskFunc function"
  return 0

def idleFunc():
  print "No specified idle function"
  return 0
  