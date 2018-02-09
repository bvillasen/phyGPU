import pycuda.driver as cuda
import sys


######################################################################
CUDA_initialized = False
def setCudaDevice( devN = None, usingAnimation=False  ):
  global CUDA_initialized
  if CUDA_initialized: return
  import pycuda.autoinit
  nDevices = cuda.Device.count()
  print "\nAvailable Devices:"
  for i in range(nDevices):
    dev = cuda.Device( i )
    print "  Device {0}: {1}".format( i, dev.name() )
  devNumber = 0
  if nDevices > 1:
    if devN == None: 
      devNumber = int(raw_input("Select device number: "))  
    else:
      devNumber = devN 
  cuda.Context.pop()  #Disable previus CUDA context
  dev = cuda.Device( devNumber )
  if usingAnimation:
    import pycuda.gl as cuda_gl
    cuda_gl.make_context(dev)
  else:
    dev.make_context()
  print "Using device {0}: {1}".format( devNumber, dev.name() ) 
  CUDA_initialized = True
  return dev

######################################################################
def mpi_setCudaDevice( pId, devN, MPI_comm, show=True ):
  cuda.init()
  if pId == 0 and show:
    nDevices = cuda.Device.count()
    print "Available Devices:"
    for i in range(nDevices):
      dev = cuda.Device( i )
      print "  Device {0}: {1}".format( i, dev.name() )
  MPI_comm.Barrier()
  #cuda.Context.pop()  #Disable previus CUDA context
  dev = cuda.Device( devN ) #device we are working on
  ctx = dev.make_context() #make a working context
  ctx.push() #let context make the lead
  print "[pId {0}] Using device {1}: {2}".format(pId, devN, ctx.get_device().name())
  return ctx, dev

#####################################################################
def getFreeMemory( show=True):
  Mbytes = float(cuda.mem_get_info()[0])/1e6
  if show:
    print " Free Global Memory: {0:.0f} MB".format(float(Mbytes))
  return cuda.mem_get_info()[0]

#####################################################################
def kernelMemoryInfo(kernel, kernelName=""):
  shared=kernel.shared_size_bytes
  regs=kernel.num_regs
  local=kernel.local_size_bytes
  const=kernel.const_size_bytes
  mbpt=kernel.max_threads_per_block
  print "=Kernel Memory=    {0}". format(kernelName)
  print("""Local:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d"""%(local,shared,regs,const,mbpt))

##################################################################### 
def np2DtoCudaArray( npArray, allowSurfaceBind=False ):
  #import pycuda.autoinit
  h, w = npArray.shape
  descr2D = cuda.ArrayDescriptor()
  descr2D.width = w
  descr2D.height = h
  descr2D.format = cuda.dtype_to_array_format(npArray.dtype)
  descr2D.num_channels = 1
  if allowSurfaceBind:
    descr.flags = cuda.array3d_flags.SURFACE_LDST
  cudaArray = cuda.Array(descr2D)
  copy2D = cuda.Memcpy2D()
  copy2D.set_src_host(npArray)
  copy2D.set_dst_array(cudaArray)
  copy2D.src_pitch = npArray.strides[0]
  copy2D.width_in_bytes = copy2D.src_pitch = npArray.strides[0]
  copy2D.src_height = copy2D.height = h
  copy2D(aligned=True)
  return cudaArray, descr2D

##################################################################### 
def np3DtoCudaArray( npArray, allowSurfaceBind=False ):
  #import pycuda.autoinit
  d, h, w = npArray.shape
  descr3D = cuda.ArrayDescriptor3D()
  descr3D.width = w
  descr3D.height = h
  descr3D.depth = d
  descr3D.format = cuda.dtype_to_array_format(npArray.dtype)
  descr3D.num_channels = 1
  descr3D.flags = 0
  if allowSurfaceBind:
    descr3D.flags = cuda.array3d_flags.SURFACE_LDST
  cudaArray = cuda.Array(descr3D)
  copy3D = cuda.Memcpy3D()
  copy3D.set_src_host(npArray)
  copy3D.set_dst_array(cudaArray)
  copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[1]
  copy3D.src_height = copy3D.height = h
  copy3D.depth = d
  copy3D()
  return cudaArray
##################################################################### 
def gpuArray3DtocudaArray( gpuArray, allowSurfaceBind=False, precision='float' ):
  #import pycuda.autoinit
  d, h, w = gpuArray.shape
  descr3D = cuda.ArrayDescriptor3D()
  descr3D.width = w
  descr3D.height = h
  descr3D.depth = d
  if precision == 'float':
    descr3D.format = cuda.dtype_to_array_format(gpuArray.dtype)
    descr3D.num_channels = 1
  elif precision == 'double': 
    descr3D.format = cuda.array_format.SIGNED_INT32
    descr3D.num_channels = 2
  else: 
    print "ERROR:  CUDA_ARRAY incompatible precision"
    sys.exit()
  descr3D.flags = 0
  if allowSurfaceBind:
    descr3D.flags = cuda.array3d_flags.SURFACE_LDST
  cudaArray = cuda.Array(descr3D)
  copy3D = cuda.Memcpy3D()
  copy3D.set_src_device(gpuArray.ptr)
  copy3D.set_dst_array(cudaArray)
  copy3D.width_in_bytes = copy3D.src_pitch = gpuArray.strides[1]
  copy3D.src_height = copy3D.height = h
  copy3D.depth = d
  copy3D()
  return cudaArray, copy3D
##################################################################### 
def gpuArray2DtocudaArray( gpuArray ):
  #import pycuda.autoinit
  h, w = gpuArray.shape
  descr2D = cuda.ArrayDescriptor()
  descr2D.width = w
  descr2D.height = h
  descr2D.format = cuda.dtype_to_array_format(gpuArray.dtype)
  descr2D.num_channels = 1
  cudaArray = cuda.Array(descr2D)
  copy2D = cuda.Memcpy2D()
  copy2D.set_src_device(gpuArray.ptr)
  copy2D.set_dst_array(cudaArray)
  copy2D.src_pitch = gpuArray.strides[0]
  copy2D.width_in_bytes = copy2D.src_pitch = gpuArray.strides[0]
  copy2D.src_height = copy2D.height = h
  copy2D(aligned=True)
  return cudaArray, copy2D