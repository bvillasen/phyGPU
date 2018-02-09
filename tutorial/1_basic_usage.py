# phyGPU basic usage
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# github: https://github.com/bvillasen
#To run you need these complementary files: CUDAheat3D.cu, volumeRender.py, CUDAvolumeRender.cu, cudaTools.py

import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
from cudaTools import  setCudaDevice, getFreeMemory

#Start the comunication with the CUDA device
