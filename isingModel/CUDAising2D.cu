// #include <pycuda-complex.hpp>
// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>


texture<int, cudaTextureType2D, cudaReadModeElementType> tex_spinsIn;
//surface<void, 2> surf_out;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


__device__ int deltaEnergy( int center, int nWidth, int nHeight, int t_i, int t_j ){
//   int center = tex2D( tex_spinsIn, t_j, t_i );
  int left   = tex2D( tex_spinsIn, t_j-1, t_i );
  int right  = tex2D( tex_spinsIn, t_j+1, t_i );
  int up     = tex2D( tex_spinsIn, t_j, t_i+1 );
  int down   = tex2D( tex_spinsIn, t_j, t_i-1 );
  
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down = tex2D( tex_spinsIn, t_j, nHeight-1 );
  if (t_i == (nHeight-1))   up = tex2D( tex_spinsIn, t_j, 0 );
  if (t_j == 0)           left = tex2D( tex_spinsIn, nWidth-1, t_i );
  if (t_j == (nWidth-1)) right = tex2D( tex_spinsIn, 0, t_i );
  
  return 2*center*(up + down + right + left );
}

__device__ int getSpinEnergy( int center, int nWidth, int nHeight, int t_i, int t_j ){
//   int center = tex2D( tex_spinsIn, t_j, t_i );
  int left   = tex2D( tex_spinsIn, t_j-1, t_i );
  int right  = tex2D( tex_spinsIn, t_j+1, t_i );
  int up     = tex2D( tex_spinsIn, t_j, t_i-1 );
  int down   = tex2D( tex_spinsIn, t_j, t_i+1 );
  
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down = tex2D( tex_spinsIn, t_j, nHeight-1 );
  if (t_i == (nHeight-1))   up = tex2D( tex_spinsIn, t_j, 0 );
  if (t_j == 0)           left = tex2D( tex_spinsIn, nWidth-1, t_i );
  if (t_j == (nWidth-1)) right = tex2D( tex_spinsIn, 0, t_i );
  
  return -center*(up + down + right + left );
}

__device__ bool metropolisAccept( int tid, float beta, int deltaE, float *randomNumbers){
  float random = randomNumbers[tid];
  float val = exp(-1*beta*deltaE);

  if ( deltaE <= 0 ) return true;
  if ( random < val ) return true;
  return false;
}

__global__ void ising_kernel( int paridad,  int nWidth, int nHeight, float beta, 
			      int *spinsOut, float *randomNumbers ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  if ( t_i%2 == paridad ) t_j +=1;
  int tid = t_j + t_i*nWidth;

  int currentSpin = tex2D( tex_spinsIn, t_j, t_i );
  int deltaE = deltaEnergy( currentSpin, nWidth, nHeight, t_i, t_j );
  if (metropolisAccept(tid, beta, deltaE, randomNumbers)) spinsOut[tid] = -1*currentSpin; 
//   if (saveEnergy) spinsEnergies[tid] = getSpinEnergy( nWidth, nHeight, t_i, t_j );
}
