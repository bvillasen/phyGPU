// #include <pycuda-complex.hpp>
#include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>


texture< int, cudaTextureType3D, cudaReadModeElementType> tex_spinsIn;
surface< void, cudaSurfaceType3D> surf_spinsOut;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


__device__ int deltaEnergy( int center, int nWidth, int nHeight, int nDepth, int t_i, int t_j, int t_k ){
  int right, left, up, down, top, bottom;
//   center = tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k);
  up =     tex3D(tex_spinsIn, (float)t_j, (float)t_i+1, (float)t_k);
  down =   tex3D(tex_spinsIn, (float)t_j, (float)t_i-1, (float)t_k);
  right =  tex3D(tex_spinsIn, (float)t_j+1, (float)t_i, (float)t_k);
  left =   tex3D(tex_spinsIn, (float)t_j-1, (float)t_i, (float)t_k);
  top =    tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k+1);
  bottom = tex3D(tex_spinsIn, (float)t_j, (float)t_i, (float)t_k-1);

  //Set PERIODIC boundary conditions
  if (t_i == 0)           down =  tex3D( tex_spinsIn, (float)t_j, (float)nHeight-1, (float)t_k );
  if (t_i == nHeight-1)   up =    tex3D( tex_spinsIn, (float)t_j, (float)0, (float)t_k );
  if (t_j == 0)           left =  tex3D( tex_spinsIn, (float)nWidth-1, (float)t_i, (float)t_k );
  if (t_j == nWidth-1)    right = tex3D( tex_spinsIn, (float)0, (float)t_i, (float)t_k );
  if (t_k == 0)           left =  tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)nDepth-1 );
  if (t_k == nDepth-1)    right = tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)0 );

  return 2*center*(up + down + right + left + top + bottom);
}

__device__ bool metropolisAccept( int tid, float beta, int deltaE, float *randomNumbers){
  float random = randomNumbers[tid];
  float val = exp(-1*beta*deltaE);

  if (deltaE<=0) return true;
  if (random < val) return true;
  return false;
}

__global__ void ising_kernel( int parity,  int nWidth, int nHeight, int nDepth, float beta,
			      int *spinsOut, float *randomNumbers,
			      float *plotData, float upVal, float downVal ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( t_i%2 == 0 ){
    if ( (t_k)%2 == parity ) t_j +=1;
  }
  else{
    if ( (t_k+1)%2 == parity ) t_j +=1;
  }

  int tid = t_j + t_i*nWidth + t_k*nWidth*blockDim.y*gridDim.y;

  int currentSpin = tex3D( tex_spinsIn, (float)t_j, (float)t_i, (float)t_k );
  int deltaE = deltaEnergy( currentSpin, nWidth, nHeight, nDepth, t_i, t_j, t_k );
  if (metropolisAccept(tid, beta, deltaE, randomNumbers)) currentSpin *= -1;

  surf3Dwrite(  currentSpin, surf_spinsOut,  t_j*sizeof(int), t_i, t_k,  cudaBoundaryModeClamp);
//   if (saveEnergy) spinsEnergies[tid] = getSpinEnergy( nWidth, nHeight, t_i, t_j );

  //Save values for plotting;
  float plotVal;
  if (currentSpin == 1 ) plotVal = upVal;
  if (currentSpin == -1 ) plotVal = downVal;
  plotData[tid] = plotVal;

}
