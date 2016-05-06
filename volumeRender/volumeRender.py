#!/usr/bin/env python
# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray


#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
myToolsDirectory = parentDirectory + "/tools"
volRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [myToolsDirectory, volRenderDirectory] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray

nWidth = 128
nHeight = 128
nDepth = 128
#nData = nWidth*nHeight*nDepth

windowTitle = "CUDA 3D volume render"


viewXmin, viewXmax = -0.5, 0.5
viewYmin, viewYmax = -0.5, 0.5
viewZmin, viewZmax = -0.5, 0.5



plotData_h = np.random.rand(nWidth*nHeight*nDepth)

def stepFunc():
  print "Default step function"

width_GL = 512*2
height_GL = 512*2

dataMax = plotData_h.max()
plotData_h = (256.*plotData_h/dataMax).astype(np.uint8).reshape(nDepth, nHeight, nWidth)
plotData_dArray = None
plotData_dArray_1 = None
transferFuncArray_d = None

viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -4.])
invViewMatrix_h = np.arange(12).astype(np.float32)
scaleX = 1.


density = 0.05
brightness = 1.0
transferOffset = 0.0
transferScale = 1.0
#linearFiltering = True
density = np.float32(density)
brightness = np.float32(brightness)
transferOffset = np.float32(transferOffset)
transferScale = np.float32(transferScale)

block2D_GL = (16, 16, 1)
grid2D_GL = (width_GL/block2D_GL[0], height_GL /block2D_GL[1] )

gl_tex = []
nTextures = 1
gl_PBO = []
#nPBOs = 1
cuda_PBO = []


frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

#CUDA device variables
c_invViewMatrix = None

#CUDA Kernels
renderKernel = None


#CUDA Textures
tex = None
transferTex = None

def computeFPS():
    global frameCount, fpsCount, fpsLimit, timer
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle(windowTitle + "      fps={0:0.2f}".format( float(ifps) ))
        fpsCount = 0

def render():
  global invViewMatrix_h, c_invViewMatrix
  global gl_PBO, cuda_PBO
  global width_GL, height_GL, density, brightness, transferOffset, transferScale
  global block2D_GL, grid2D_GL
  global tex, transferTex
  global testData_d
  cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_h)
  for i in range(nTextures):
    if i == 0:
      brightness = np.float32(1.0)
      tex.set_array(plotData_dArray)
    if i == 1:
      brightness = np.float32(2)
      tex.set_array(plotData_dArray_1)
    # map PBO to get CUDA device pointer
    cuda_PBO_map = cuda_PBO[i].map()
    cuda_PBO_ptr, cuda_PBO_size = cuda_PBO_map.device_ptr_and_size()
    cuda.memset_d32( cuda_PBO_ptr, 0, width_GL*height_GL )
    renderKernel( np.intp(cuda_PBO_ptr), np.int32(width_GL), np.int32(height_GL), density, brightness, transferOffset, transferScale, grid=grid2D_GL, block = block2D_GL, texrefs=[tex, transferTex] )
    cuda_PBO_map.unmap()

def display():
  global viewRotation, viewTranslation, invViewMatrix_h
  global timer

  timer = time.time()
  stepFunc()

  modelView = np.ones(16)
  glMatrixMode(GL_MODELVIEW)
  glPushMatrix()
  glLoadIdentity()
  glRotatef(-viewRotation[0], 1.0, 0.0, 0.0)
  glRotatef(-viewRotation[1], 0.0, 1.0, 0.0)
  glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2])
  modelView = glGetFloatv(GL_MODELVIEW_MATRIX )
  modelView = modelView.reshape(16).astype(np.float32)
  glPopMatrix()
  #invViewMatrix_h = modelView[:12]
  invViewMatrix_h[0] = modelView[0]/scaleX
  invViewMatrix_h[1] = modelView[4]
  invViewMatrix_h[2] = modelView[8]
  invViewMatrix_h[3] = modelView[12]
  invViewMatrix_h[4] = modelView[1]
  invViewMatrix_h[5] = modelView[5]
  invViewMatrix_h[6] = modelView[9]
  invViewMatrix_h[7] = modelView[13]
  invViewMatrix_h[8] = modelView[2]
  invViewMatrix_h[9] = modelView[6]
  invViewMatrix_h[10] = modelView[10]
  invViewMatrix_h[11] = modelView[14]
  render()
   # display results
  glClear(GL_COLOR_BUFFER_BIT)

  for i in range(nTextures):

    # draw image from PBO
    #glDisable(GL_DEPTH_TEST)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # draw using texture
    # copy from pbo to texture
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER, gl_PBO[i])
    glBindTexture(GL_TEXTURE_2D, gl_tex[i])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_GL, height_GL, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0)
    # draw textured quad
    #glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    if nTextures == 2:
      if i==0: glVertex2f(-1., -0.5)
      if i==1: glVertex2f( 0., -0.5)
    else:  glVertex2f(-0.5, -0.5)
    glTexCoord2f(1, 0)
    if nTextures == 2:
      if i==0: glVertex2f( 0., -0.5)
      if i==1: glVertex2f( 1., -0.5)
    else: glVertex2f(0.5, -0.5)
    glTexCoord2f(1, 1)
    if nTextures == 2:
      if i==0: glVertex2f( 0., 0.5)
      if i==1: glVertex2f( 1., 0.5)
    else: glVertex2f(0.5, 0.5)
    glTexCoord2f(0, 1)
    if nTextures == 2:
      if i==0: glVertex2f(-1., 0.5)
      if i==1: glVertex2f( 0., 0.5)
    else: glVertex2f(-0.5, 0.5)
    glEnd()

    #glDisable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

  glutSwapBuffers();
  timer = time.time() - timer
  computeFPS()


def iDivUp( a, b ):
  if a%b != 0:
    return a/b + 1
  else:
    return a/b



GL_initialized = False
def initGL():
  global GL_initialized
  if GL_initialized: return
  glutInit()
  glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE )
  glutInitWindowSize(width_GL*nTextures, height_GL)
  #glutInitWindowPosition(50, 50)
  glutCreateWindow( windowTitle )
  #glew.glewInit()
  GL_initialized = True
  print "OpenGL initialized"

def initPixelBuffer():
  global cuda_PBO
  for i in range(nTextures):
    PBO = glGenBuffers(1)
    gl_PBO.append(PBO)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, gl_PBO[i])
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER, width_GL*height_GL*4, None, GL_STREAM_DRAW_ARB)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0)
    cuda_PBO.append( cuda_gl.RegisteredBuffer(long(gl_PBO[i])) )
  #print "Buffer Created"
  #Create texture which we use to display the result and bind to gl_tex
  glEnable(GL_TEXTURE_2D)

  for i in range(nTextures):
    tex = glGenTextures(1)
    gl_tex.append( tex )
    glBindTexture(GL_TEXTURE_2D, gl_tex[i])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_GL, height_GL, 0,
		  GL_RGBA, GL_UNSIGNED_BYTE, None);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
  #print "Texture Created"

def initCUDA():
  global plotData_dArray
  global tex, transferTex
  global transferFuncArray_d
  global c_invViewMatrix
  global renderKernel
  #print "Compiling CUDA code for volumeRender"
  cudaCodeFile = open(volRenderDirectory + "/CUDAvolumeRender.cu","r")
  cudaCodeString = cudaCodeFile.read()
  cudaCodeStringComplete = cudaCodeString
  cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True, include_dirs=[volRenderDirectory] )
  tex = cudaCode.get_texref("tex")
  transferTex = cudaCode.get_texref("transferTex")
  c_invViewMatrix = cudaCode.get_global('c_invViewMatrix')[0]
  renderKernel = cudaCode.get_function("d_render")

  if not plotData_dArray: plotData_dArray = np3DtoCudaArray( plotData_h )
  tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  tex.set_filter_mode(cuda.filter_mode.LINEAR)
  tex.set_address_mode(0, cuda.address_mode.CLAMP)
  tex.set_address_mode(1, cuda.address_mode.CLAMP)
  tex.set_array(plotData_dArray)

  transferFunc = np.array([
    [  0.0, 0.0, 0.0, 0.0, ],
    [  1.0, 0.0, 0.0, 1.0, ],
    [  1.0, 0.5, 0.0, 1.0, ],
    [  1.0, 1.0, 0.0, 1.0, ],
    [  0.0, 1.0, 0.0, 1.0, ],
    [  0.0, 1.0, 1.0, 1.0, ],
    [  0.0, 0.0, 1.0, 1.0, ],
    [  1.0, 0.0, 1.0, 1.0, ],
    [  0.0, 0.0, 0.0, 0.0, ]]).astype(np.float32)
  transferFuncArray_d, desc = np2DtoCudaArray( transferFunc )
  transferTex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  transferTex.set_filter_mode(cuda.filter_mode.LINEAR)
  transferTex.set_address_mode(0, cuda.address_mode.CLAMP)
  transferTex.set_address_mode(1, cuda.address_mode.CLAMP)
  transferTex.set_array(transferFuncArray_d)
  print "CUDA volumeRender initialized\n"


def keyboard(*args):
  global transferScale, brightness, density, transferOffset
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()
  if args[0] == '1':
    transferScale += np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '2':
    transferScale -= np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '4':
    brightness -= np.float32(0.1)
    print "Image Brightness : ",brightness
  if args[0] == '5':
    brightness += np.float32(0.1)
    print "Image Brightness : ",brightness
  if args[0] == '7':
    density -= np.float32(0.01)
    print "Image Density : ",density
  if args[0] == '8':
    density += np.float32(0.01)
    print "Image Density : ",density
  if args[0] == '3':
    transferOffset += np.float32(0.01)
    print "Image Offset : ", transferOffset
  if args[0] == '6':
    transferOffset -= np.float32(0.01)
    print "Image Offset : ", transferOffset


def specialKeys( key, x, y ):
  if key==GLUT_KEY_UP:
    print "UP-arrow pressed"
  if key==GLUT_KEY_DOWN:
    print "DOWN-arrow pressed"


ox = 0
oy = 0
buttonState = 0
def mouse(button, state, x , y):
  global ox, oy, buttonState

  if state == GLUT_DOWN:
    buttonState |= 1<<button
    if button == 3:  #wheel up
      viewTranslation[2] += 0.5
    if button == 4:  #wheel down
      viewTranslation[2] -= 0.5
  elif state == GLUT_UP:
    buttonState = 0
  ox = x
  oy = y
  glutPostRedisplay()

def motion(x, y):
  global viewRotation, viewTranslation
  global ox, oy, buttonState
  dx = x - ox
  dy = y - oy
  if buttonState == 4:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 2:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 1:
    viewRotation[0] += dy/5.
    viewRotation[1] += dx/5.
  ox = x
  oy = y
  glutPostRedisplay()

def reshape(w, h):
  global width_GL, height_GL
  global grid2D_GL, block2D_GL
  initPixelBuffer()
  #width_GL, height_GL = w, h
  grid2D_GL = ( iDivUp(width_GL, block2D_GL[0]), iDivUp(height_GL, block2D_GL[1]) )
  glViewport(0, 0, w, h)

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  #glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
  if w <= h: glOrtho( viewXmin, viewXmax,
				viewYmin*h/w, viewYmax*h/w,
				viewZmin, viewZmax)
  else: glOrtho(viewXmin*w/h, viewXmax*w/h,
		viewYmin, viewYmax,
		viewZmin, viewZmax)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()


def startGL():
  glutDisplayFunc(display)
  glutKeyboardFunc(keyboard)
  glutSpecialFunc(specialKeys)
  glutMouseFunc(mouse)
  glutMotionFunc(motion)
  glutReshapeFunc(reshape)
  glutIdleFunc(glutPostRedisplay)
  glutMainLoop()

#OpenGL main
def animate():
  global windowTitle
  print "Starting Volume Render"
  #initGL()
  #import pycuda.gl.autoinit
  initCUDA()
  initPixelBuffer()
  startGL()
