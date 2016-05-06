#include <pycuda-complex.hpp>
#include <pycuda-helpers.hpp>
#define pi 3.14159265f
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef   pycuda::complex<cudaP> pyComplex;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getAlphas_kernel( cudaP dx, cudaP dy, cudaP dz, cudaP xMin, cudaP yMin, cudaP zMin,
				  cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
				  pyComplex *psi1, cudaP *alphas){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  cudaP g = 8000;
  cudaP psi_mod = abs(psi1[tid]);
  cudaP result = (cudaP(0.5)*( gammaX*x*x + gammaY*y*y + gammaZ*z*z )) + g*psi_mod*psi_mod;
  alphas[tid] = result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getPartialsXY_kernel( cudaP Lx, cudaP Ly, pyComplex *fftTrnf,
				      pyComplex *partialxfft, cudaP *kxfft,
				      pyComplex *partialyfft, cudaP *kyfft ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP kx = kxfft[t_j];
  cudaP ky = kyfft[t_i];
  pyComplex i_complex( cudaP(0.), cudaP(1.0));
  pyComplex psiFFT = fftTrnf[tid];
  
  partialxfft[tid] = kx*i_complex*psiFFT;
  partialyfft[tid] = ky*i_complex*psiFFT;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getFFTderivatives_kernel( cudaP Lx, cudaP Ly, cudaP Lz, pyComplex *fftTrnf,
				      cudaP *kxfft, cudaP *kyfft, cudaP *kzfft,
				      pyComplex *partialxfft, pyComplex *partialyfft, pyComplex *laplacianfft){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP kx = kxfft[t_j];
  cudaP ky = kyfft[t_i];
  cudaP kz = kzfft[t_k];
  cudaP k2 = kx*kx + ky*ky + kz*kz;
  pyComplex i_complex( cudaP(0.), cudaP(1.0));
  pyComplex psiFFT = fftTrnf[tid];
  
  partialxfft[tid] = kx*i_complex*psiFFT;
  partialyfft[tid] = ky*i_complex*psiFFT;
  laplacianfft[tid] = -k2 * psiFFT;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void setBoundryConditions_kernel(int nWidth, int nHeight, int nDepth, pycuda::complex<cudaP> *psi){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  if ( t_i==0 or t_i==(nHeight-1) or t_j==0 or t_j==(nWidth-1) or t_k==0 or t_k==(nDepth-1)){
    psi[tid] = 0;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void implicitStep1_kernel( cudaP xMin, cudaP yMin, cudaP zMin, cudaP dx, cudaP dy, cudaP dz,  
			     cudaP alpha, cudaP omega, cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
			     pyComplex *partialX_d,
			     pyComplex *partialY_d,
			     pyComplex *psi1_d, pyComplex *G1_d,
			     cudaP x0,cudaP y0){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  pyComplex iComplex( cudaP(0.0), cudaP(1.0) );
  pyComplex complex1( cudaP(1.0), cudaP(0.0) );
  pyComplex psi1, partialX, partialY, Vtrap, torque, lz, result;
  cudaP g = 8000;
  
  psi1 = psi1_d[tid];
  cudaP psiMod = abs(psi1);
  Vtrap = psi1*(gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5);
  torque = psi1*g*psiMod*psiMod;
  partialX = partialX_d[tid];
  partialY = partialY_d[tid];
  lz = iComplex * omega * (partialY*(x-x0) - partialX*(y-y0)); 
  G1_d[tid] = psi1*alpha - Vtrap - torque - lz;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void implicitStep2_kernel( cudaP dt, cudaP *kx, cudaP *ky, cudaP *kz, cudaP alpha, 
			      pyComplex *psiTransf, pyComplex *GTranf){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP kX = kx[t_j];
  cudaP kY = ky[t_i];
  cudaP kZ = kz[t_k];
  cudaP k2 = kX*kX + kY*kY + kZ*kZ;
  
  pyComplex factor, timeStep, psiT, Gt;
  factor = cudaP(2.0) / ( 2 + dt*(k2 + 2*alpha) );
  psiT = psiTransf[tid];
  Gt = GTranf[tid];
  psiTransf[tid] = factor * ( psiT + Gt*dt); 
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void findActivity_kernel( cudaP minDensity, pyComplex *psi_d, unsigned char *activity ){
  int t_j = blockIdx.x* %(blockDim.x)s  + threadIdx.x;
  int t_i = blockIdx.y* %(blockDim.y)s  + threadIdx.y;
  int t_k = blockIdx.z* %(blockDim.z)s  + threadIdx.z;
  int tid = t_j + t_i* %(blockDim.x)s * %(gridDim.x)s  + t_k* %(blockDim.x)s * %(gridDim.x)s * %(blockDim.y)s * %(gridDim.y)s ;
  int tid_b = threadIdx.x + threadIdx.y* %(blockDim.x)s  + threadIdx.z* %(blockDim.x)s * %(blockDim.y)s ;
//   int bid = blockIdx.x + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s ;
  
  pyComplex psi = psi_d[tid];
  __shared__ cudaP density[ %(THREADS_PER_BLOCK)s ];
  density[tid_b] = norm(psi);
  __syncthreads();
  
  int i =  %(blockDim.x)s * %(blockDim.y)s * %(blockDim.z)s  / 2;
  while ( i > 0 ){
    if ( tid_b < i ) density[tid_b] = density[tid_b] + density[tid_b+i];
    __syncthreads();
    i /= 2;
  }
  if ( tid_b == 0 ){
    if (density[0] >= minDensity ) {
      activity[ blockIdx.x + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //right 
      if (blockIdx.x <  %(gridDim.x)s -1) activity[ (blockIdx.x+1) + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //left
      if (blockIdx.x > 0) activity[ (blockIdx.x-1) + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //up 
      if (blockIdx.y <  %(gridDim.y)s -1) activity[ blockIdx.x + (blockIdx.y+1)* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //down
      if (blockIdx.y > 0) activity[ blockIdx.x + (blockIdx.y-1)* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //top 
      if (blockIdx.z <  %(gridDim.z)s -1) activity[ blockIdx.x + blockIdx.y* %(gridDim.x)s  + (blockIdx.z+1)* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
      //bottom
      if (blockIdx.z > 0) activity[ blockIdx.x + blockIdx.y* %(gridDim.x)s  + (blockIdx.z-1)* %(gridDim.x)s * %(gridDim.y)s  ] = (unsigned char) 1;
    }
  }
}
__global__ void getActivity_kernel( cudaP *psiOther, unsigned char *activity ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.z;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  
  if ( activeBlock ) psiOther[tid] = cudaP(0.4);
  else psiOther[tid] = cudaP(0.08);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getVelocity_kernel( int neighbors, cudaP dx, cudaP dy, cudaP dz, pyComplex *psi, unsigned char *activity, cudaP *psiOther){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z ==  %(gridDim.z)s  -1 ) ) return; 
 
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  if ( !activeBlock ) return; 
  pyComplex center = psi[tid];
  __shared__ pyComplex psi_sh[ %(B_WIDTH)s ][ %(B_HEIGHT)s ][ %(B_DEPTH)s ];
  __syncthreads();
  psi_sh[threadIdx.x][threadIdx.y][threadIdx.z] = center;
  __syncthreads();
    
  cudaP dxInv = cudaP(1.0)/dx;
  cudaP dyInv = cudaP(1.0)/dy;
  cudaP dzInv = cudaP(1.0)/dz;
  pyComplex gradient_x, gradient_y, gradient_z;
    

  if ( threadIdx.x == 0 ) gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - center )*dxInv;
  else if ( threadIdx.x == (blockDim.x-1) ) gradient_x = ( center - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] )*dxInv;
  else gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] ) * dxInv* cudaP(0.5);

  if ( threadIdx.y == 0 ) gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - center )*dyInv;
  else if ( threadIdx.y == (blockDim.y-1) ) gradient_y = ( center - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] )*dyInv;
  else gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] ) * dyInv* cudaP(0.5);
  
  if ( threadIdx.z == 0 ) gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - center )*dzInv;
  else if ( threadIdx.z == (blockDim.z-1) ) gradient_z = ( center - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] )*dzInv;
  else gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] ) * dzInv* cudaP(0.5);

//   __syncthreads();
  cudaP rho = norm(center) + cudaP(5e-6);
  cudaP velX = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/rho;
  cudaP velY = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/rho;
  cudaP velZ = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/rho; 

  psiOther[tid] =  sqrt( velX*velX + velY*velY + velZ*velZ ) ;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ pyComplex vortexCore( const int nWidth, const int nHeight, const int nDepth, 
				 const cudaP xMin, const cudaP yMin, const cudaP zMin, 
				 const cudaP dx, const cudaP dy, const cudaP dz, 
				 const int t_i, const int t_j, const int t_k, const int tid, const int tid_b,
				 const cudaP gammaX, const cudaP gammaY, const cudaP gammaZ, 
				 const cudaP omega,  pyComplex *psiStep){
  pyComplex center = psiStep[tid];
  __shared__ pyComplex psi_sh[ %(B_WIDTH)s ][ %(B_HEIGHT)s ][ %(B_DEPTH)s ];
  __syncthreads();
  psi_sh[threadIdx.x][threadIdx.y][threadIdx.z] = center;
  __syncthreads();
  
  const cudaP dxInv = cudaP(1.0)/dx;
  const cudaP dyInv = cudaP(1.0)/dy;
  const cudaP dzInv = cudaP(1.0)/dz;
  const cudaP x = t_j*dx + xMin;
  const cudaP y = t_i*dy + yMin;
  const cudaP z = t_k*dz + zMin;
  
  const pyComplex iComplex( cudaP(0.), cudaP(1.) );
//   pyComplex laplacian( cudaP(0.), cudaP(0.) );
//   pyComplex Lz( cudaP(0.), cudaP(0.) );
  pyComplex psiMinus, psiPlus;
  
  // laplacian X-term
  if (threadIdx.x==0 ){
    psiMinus = psiStep[ (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
  }
  else if (threadIdx.x==blockDim.x-1){
    psiPlus  = psiStep[ (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
    psiMinus = psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];
  }
//   psiMinus = threadIdx.x == 0 ? 
// 	      psiStep[ (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] : 
// 	      psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];
//   psiPlus  = threadIdx.x == (nWidth-1) ? 
// 	      psiStep[ (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] : 
// 	      psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
  pyComplex laplacian = ( psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;
  pyComplex Lz =  -iComplex*( psiPlus - psiMinus )*dxInv*cudaP(0.5)*y;
  
  // laplacian Y-term
  if (threadIdx.y==0 ){
    psiMinus = psiStep[ t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z];
  }
  else if (threadIdx.y==blockDim.y-1){
    psiPlus  = psiStep[ t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z];
  }
  laplacian += ( psiPlus + psiMinus - cudaP(2)*center )*dyInv*dyInv;
  Lz +=  iComplex*( psiPlus - psiMinus)*dyInv*cudaP(0.5)*x;
  
  // laplacian Z-term
  if (threadIdx.z==0 ){
    psiMinus = psiStep[ t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1];
  }
  else if (threadIdx.z==blockDim.z-1){
    psiPlus  = psiStep[ t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1];
  }
  laplacian += ( psiPlus + psiMinus - cudaP(2)*center )*dzInv*dzInv; 
  
  
//   pyComplex  GP; 
  cudaP Vtrap_GP = (gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5) + 8000*norm(center);
//   cudaP GP = ;  
  
  return iComplex*(laplacian*cudaP(0.5) - Vtrap_GP*center - Lz*omega);
//   return iComplex*(laplacian*cudaP(0.5) - Vtrap - GP );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ pyComplex vortexCore_fft( int nWidth, int nHeight, int nDepth, cudaP xMin, cudaP yMin, cudaP zMin, 
				 cudaP dx, cudaP dy, cudaP dz, int t_i, int t_j, int t_k, int tid,
				 cudaP gammaX, cudaP gammaY, cudaP gammaZ, cudaP omega, cudaP x0, cudaP y0, 
				 pyComplex *psiStep, pyComplex *laplacian, pyComplex *partialX, pyComplex *partialY){
  pyComplex center = psiStep[tid];
//   cudaP dxInv = cudaP(1.0)/dx;
//   cudaP dyInv = cudaP(1.0)/dy;
//   cudaP dzInv = cudaP(1.0)/dz;
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  pyComplex iComplex( cudaP(0.), cudaP(1.) );
  pyComplex Vtrap, GP, Lz; 
  Vtrap = (gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5)*center;
  GP = cudaP(8000.)*norm(center)*center;
  Lz = iComplex*( partialY[tid] - partialX[tid] )*omega;
  
  return iComplex * ( laplacian[tid]*cudaP(0.5) - Vtrap - GP - Lz );

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_psiReal;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_psiImag;
surface< void, cudaSurfaceType3D> surf_psiReal;
surface< void, cudaSurfaceType3D> surf_psiImag;
__device__ pyComplex vortexCore_tex
	    ( cudaP xMin, cudaP yMin, cudaP zMin, 
	      cudaP dx, cudaP dy, cudaP dz, int t_i, int t_j, int t_k, 
	      cudaP gammaX, cudaP gammaY, cudaP gammaZ, cudaP omega ){
  

  const cudaP dxInv = 1.0f/dx;
//   cudaP dyInv = 1.0f/dy;
//   cudaP dzInv = 1.0f/dz;
  const cudaP x = t_j*dx + xMin;
  const cudaP y = t_i*dy + yMin;
  const cudaP z = t_k*dz + zMin;
  
  const pyComplex iComplex( 0, 1.0f );
  pyComplex center, psiPlus, psiMinus, laplacian;
  pyComplex Lz;
  center._M_re =    fp_tex3D(tex_psiReal, t_j, t_i, t_k);
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j+1, t_i, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j-1, t_i, t_k);
  
  center._M_im =    fp_tex3D(tex_psiImag, t_j, t_i, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j+1, t_i, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j-1, t_i, t_k);

  laplacian = (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;
  Lz = iComplex*( psiMinus - psiPlus )*dxInv*cudaP(0.5)*y ;
  
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j, t_i+1, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i-1, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j, t_i+1, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i-1, t_k);
  laplacian += (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;
  Lz +=  iComplex*( (psiPlus - psiMinus)*dxInv*cudaP(0.5)*x );
  
  psiPlus._M_re =   fp_tex3D(tex_psiReal, t_j, t_i, t_k+1);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i, t_k-1);
  psiPlus._M_im =   fp_tex3D(tex_psiImag, t_j, t_i, t_k+1);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i, t_k-1);
  laplacian +=  (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;

  const cudaP Vtrap_GP = 8000*norm(center) + (gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5); 

  return iComplex*(laplacian*cudaP(0.5) - (Vtrap_GP)*center - Lz*omega);
//   return iComplex*(laplacian*cudaP(0.5) - (Vtrap_GP)*center );
}
__global__ void getVelocity_tex_kernel(  cudaP dx, cudaP dy, cudaP dz, unsigned char *activity, cudaP *psiOther){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z ==  %(gridDim.z)s  -1 ) ) return; 
 
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  if ( !activeBlock ) return; 
  
  cudaP dxInv = cudaP(1.0)/dx;
  cudaP dyInv = cudaP(1.0)/dy;
  cudaP dzInv = cudaP(1.0)/dz;
  pyComplex gradient_x, gradient_y, gradient_z, center, psiPlus, psiMinus;

  center._M_re =    fp_tex3D(tex_psiReal, t_j, t_i, t_k);
  center._M_im =    fp_tex3D(tex_psiImag, t_j, t_i, t_k);

  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j+1, t_i, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j+1, t_i, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j-1, t_i, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j-1, t_i, t_k);
  gradient_x = ( psiPlus - psiMinus)*dxInv*cudaP(0.5);
  
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j, t_i+1, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j, t_i+1, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i-1, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i-1, t_k);
  gradient_y = ( psiPlus - psiMinus)*dyInv*cudaP(0.5);
  
  psiPlus._M_re =   fp_tex3D(tex_psiReal, t_j, t_i, t_k+1);
  psiPlus._M_im =   fp_tex3D(tex_psiImag, t_j, t_i, t_k+1);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i, t_k-1);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i, t_k-1);
  gradient_z = ( psiPlus - psiMinus)*dzInv*cudaP(0.5);

  cudaP rho = norm(center) + cudaP(5e-6);
  cudaP velX = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/rho;
  cudaP velY = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/rho;
  cudaP velZ = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/rho; 

  psiOther[tid] =  sqrt( velX*velX + velY*velY + velZ*velZ ) ;
}
////////////////////////////////////////////////////////////////////////////////
//////////////////////           EULER                //////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void eulerStep_kernel( const int nWidth, const int nHeight, const int nDepth, 
				  const cudaP slopeCoef, const cudaP weight, 
				  const cudaP xMin, const cudaP yMin, const cudaP zMin, 
				  const cudaP dx, const cudaP dy, const cudaP dz, const cudaP dt, 
				  const cudaP gammaX, const cudaP gammaY, const cudaP gammaZ, const cudaP omega,
				      pyComplex *psi_d, pyComplex *psiStepIn, pyComplex *psiStepOut, pyComplex *psiRunge,
				      unsigned char lastRK4Step, unsigned char *activity ){
  const int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  const int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  const int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  const int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  const int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z == gridDim.z -1 ) ) return; 
  //Unactive blocks are skiped
  __shared__ unsigned char activeBlock;
  if ( tid_b == 0 ) activeBlock = activity[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y];
  __syncthreads();
  if ( !activeBlock ) return;
  
  pyComplex value;
  value = vortexCore( nWidth, nHeight, nDepth, xMin, yMin, zMin, 
		      dx, dy, dz, t_i, t_j, t_k, tid, tid_b,
		      gammaX, gammaY, gammaZ, omega, psiStepIn );
  value = dt*value;
  
  if (lastRK4Step ){
    value = psiRunge[tid] + slopeCoef*value/cudaP(6.); 
    psiRunge[tid] = value;
    psiStepOut[tid] = value;
    psi_d[tid] = value;
  }  
  else{
    psiStepOut[tid] = psi_d[tid] + weight*value;
    //add to rk4 final value
    psiRunge[tid] = psiRunge[tid] + slopeCoef*value/cudaP(6.);
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void eulerStep_texture_kernel(  const cudaP slopeCoef, const cudaP weight, 
				  const cudaP xMin, const cudaP yMin, const cudaP zMin, 
				  const cudaP dx, const cudaP dy, const cudaP dz, const cudaP dt, 
				  const cudaP gammaX, const cudaP gammaY, const cudaP gammaZ, const cudaP omega,
				      pyComplex *psi_d, pyComplex *psiRunge,
				      unsigned char lastRK4Step, unsigned char *activity ){
  const int t_j = blockIdx.x* %(blockDim.x)s  + threadIdx.x;
  const int t_i = blockIdx.y* %(blockDim.y)s  + threadIdx.y;
  const int t_k = blockIdx.z* %(blockDim.z)s  + threadIdx.z;
  const int tid = t_j + t_i* %(blockDim.x)s * %(gridDim.x)s  + t_k* %(blockDim.x)s * %(gridDim.x)s * %(blockDim.y)s * %(gridDim.y)s ;
  const int tid_b = threadIdx.x + threadIdx.y* %(blockDim.x)s  + threadIdx.z* %(blockDim.x)s * %(blockDim.y)s ;
//   int bid = blockIdx.x + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s ;
  
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or  %(blockDim.x)s  ==  %(gridDim.x)s  -1 ) or ( blockIdx.y == 0 or  %(blockDim.y)s  ==  %(gridDim.y)s  -1 ) or ( blockIdx.z == 0 or  %(blockDim.z)s  ==  %(gridDim.z)s  -1 ) ) return; 
  //Unactive blocks are skiped
  __shared__ unsigned char activeBlock;
  if ( tid_b == 0 ) activeBlock = activity[blockIdx.x + blockIdx.y* %(gridDim.x)s  + blockIdx.z* %(gridDim.x)s * %(gridDim.y)s ];
  __syncthreads();
  if ( !activeBlock ) return;
  
  pyComplex value;
  value = vortexCore_tex( xMin, yMin, zMin, 
		      dx, dy, dz, t_i, t_j, t_k,
		      gammaX, gammaY, gammaZ, omega );
  value = dt*value;
  
  if (lastRK4Step ){
    value = psiRunge[tid] + slopeCoef*value/cudaP(6.); 
    psiRunge[tid] = value;
    psi_d[tid] = value;
    surf3Dwrite(  value._M_re, surf_psiReal,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
    surf3Dwrite(  value._M_im, surf_psiImag,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);    
  }  
  else{
    //add to rk4 final value
    psiRunge[tid] = psiRunge[tid] + slopeCoef*value/cudaP(6.);
    value = psi_d[tid] + weight*value;
    surf3Dwrite(  value._M_re, surf_psiReal,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
    surf3Dwrite(  value._M_im, surf_psiImag,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
   
    
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void eulerStep_fft_kernel( int nWidth, int nHeight, int nDepth, cudaP slopeCoef, cudaP weight, 
				      cudaP xMin, cudaP yMin, cudaP zMin, cudaP dx, cudaP dy, cudaP dz, cudaP dt, 
				      cudaP gammaX, cudaP gammaY, cudaP gammaZ, cudaP x0, cudaP y0, cudaP omega,
				      pyComplex *psi_d, pyComplex *psiStepIn, pyComplex *psiStepOut, pyComplex *psiRunge,
				      pyComplex *laplacian, pyComplex *partialX, pyComplex *partialY,
				      unsigned char lastRK4Step, unsigned char *activity ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z == gridDim.z -1 ) ) return; 
  //Unactive blocks are skiped
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y];
  __syncthreads();
  if ( !activeBlock ) return;
  
  pyComplex value;
  value = vortexCore_fft( nWidth, nHeight, nDepth, xMin, yMin, zMin, 
		      dx, dy, dz, t_i, t_j, t_k, tid, 
		      gammaX, gammaY, gammaZ, omega, x0, y0, 
		      psiStepIn, laplacian, partialX, partialY  );
  value = dt*value;
  
  if (lastRK4Step ){
    pyComplex valueOut = psiRunge[tid] + slopeCoef*value/cudaP(6.); 
    psiRunge[tid] = valueOut;
    psiStepOut[tid] = valueOut;
    psi_d[tid] = valueOut;
  }  
  else{
    psiStepOut[tid] = psi_d[tid] + weight*value;
    //add to rk4 final value
    psiRunge[tid] = psiRunge[tid] + slopeCoef*value/cudaP(6.);
  }
}
