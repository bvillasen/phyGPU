import sys, time, os
import numpy as np
import pycuda.driver as cuda
from cudaTools import  setCudaDevice, getFreeMemory
