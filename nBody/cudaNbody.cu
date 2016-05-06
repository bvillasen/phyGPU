#include "vector3D.hT"


extern "C"{
  
__global__ void getAccel_kernel( const int nParticles, 
			     const int usingAnimation, 
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *accelX, cudaP *accelY, cudaP *accelZ,
			     cudaP dt, cudaP epsilon, float *cuda_VOB ){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (tid < nParticles){
    
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(TPB)s ];
    __shared__ cudaP posY_sh[ %(TPB)s ];
    __shared__ cudaP posZ_sh[ %(TPB)s ];
    __syncthreads();
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    Vector3D accel( 0., 0., 0. );
    Vector3D posOther, deltaPos;
    cudaP distInv;
    int idOther;
    for ( int blockNumber=0; blockNumber<gridDim.x; blockNumber++ ){
      idOther =  blockNumber*blockDim.x + threadIdx.x;
      posX_sh[ threadIdx.x ] = posX[ idOther ];
      posY_sh[ threadIdx.x ] = posY[ idOther ];
      posZ_sh[ threadIdx.x ] = posZ[ idOther ];
      __syncthreads();
      
      idOther = threadIdx.x;
      for ( int i=0; i<blockDim.x; i++ ){
	if (idOther >= blockDim.x ) idOther = 0;
	posOther.redefine( posX_sh[ idOther ], posY_sh[ idOther ], posZ_sh[ idOther ] );
	deltaPos = posOther - pos;
	distInv = rsqrt( deltaPos.norm2() + epsilon );
	accel += deltaPos*(distInv*distInv*distInv );
	idOther++;
      }
    }
//     accel *= Gmass;
    

    velX[tid] = vel.x + 0.5*dt*( accelX[tid] + accel.x );
    velY[tid] = vel.y + 0.5*dt*( accelY[tid] + accel.y );
    velZ[tid] = vel.z + 0.5*dt*( accelZ[tid] + accel.z );

    
    accelX[tid] = accel.x;
    accelY[tid] = accel.y;
    accelZ[tid] = accel.z;
        
    
    //Save data in animation buffer
    if ( usingAnimation ){
      cuda_VOB[3*tid + 0] = float(pos.x);
      cuda_VOB[3*tid + 1] = float(pos.y);
      cuda_VOB[3*tid + 2] = float(pos.z);
    }
  }
}
  
__global__ void getAccel_kernel_mpi( const int nParticles, 
			     const int usingAnimation, 
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *posAllX, cudaP *posAllY, cudaP *posAllZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *accelX, cudaP *accelY, cudaP *accelZ,
			     cudaP dt, cudaP epsilon, const int nProc,  float *cuda_VOB ){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (tid < nParticles){
    
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(TPB)s ];
    __shared__ cudaP posY_sh[ %(TPB)s ];
    __shared__ cudaP posZ_sh[ %(TPB)s ];
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    Vector3D accel( 0., 0., 0. );
    Vector3D posOther, deltaPos;
    cudaP distInv;
    int idOther;
    for ( int blockNumber=0; blockNumber<gridDim.x*nProc; blockNumber++ ){
      idOther =  blockNumber*blockDim.x + threadIdx.x;
      posX_sh[ threadIdx.x ] = posAllX[ idOther ];
      posY_sh[ threadIdx.x ] = posAllY[ idOther ];
      posZ_sh[ threadIdx.x ] = posAllZ[ idOther ];
      __syncthreads();
      
      idOther = threadIdx.x;
      for ( int i=0; i<blockDim.x; i++ ){
	if (idOther >= blockDim.x ) idOther = 0;
	posOther.redefine( posX_sh[ idOther ], posY_sh[ idOther ], posZ_sh[ idOther ] );
	deltaPos = posOther - pos;
	distInv = rsqrt( deltaPos.norm2() + epsilon );
	accel += deltaPos*(distInv*distInv*distInv );
	idOther++;
      }
    }
//     accel *= Gmass;
    

    velX[tid] = vel.x + 0.5*dt*( accelX[tid] + accel.x );
    velY[tid] = vel.y + 0.5*dt*( accelY[tid] + accel.y );
    velZ[tid] = vel.z + 0.5*dt*( accelZ[tid] + accel.z );

    
    accelX[tid] = accel.x;
    accelY[tid] = accel.y;
    accelZ[tid] = accel.z;
        
    
    //Save data in animation buffer
    if ( usingAnimation ){
      cuda_VOB[3*tid + 0] = float(pos.x);
      cuda_VOB[3*tid + 1] = float(pos.y);
      cuda_VOB[3*tid + 2] = float(pos.z);
    }
  }
}


__global__ void getEnergy_kernel( const int nParticles, const cudaP G, const cudaP mass, 
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *posAllX, cudaP *posAllY, cudaP *posAllZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *energyAll, cudaP epsilon, const int nProc ){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (tid < nParticles){
    
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(TPB)s ];
    __shared__ cudaP posY_sh[ %(TPB)s ];
    __shared__ cudaP posZ_sh[ %(TPB)s ];
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    Vector3D posOther, deltaPos;
    cudaP energy = 0;
    int idOther;
    for ( int blockNumber=0; blockNumber<gridDim.x*nProc; blockNumber++ ){
      idOther =  blockNumber*blockDim.x + threadIdx.x;
      posX_sh[ threadIdx.x ] = posAllX[ idOther ];
      posY_sh[ threadIdx.x ] = posAllY[ idOther ];
      posZ_sh[ threadIdx.x ] = posAllZ[ idOther ];
      __syncthreads();
    
      idOther = threadIdx.x;
      for ( int i=0; i<blockDim.x; i++ ){
	if (idOther >= blockDim.x ) idOther = 0;
	posOther.redefine( posX_sh[ idOther ], posY_sh[ idOther ], posZ_sh[ idOther ] );
	deltaPos = posOther - pos;
	energy -= rsqrt( deltaPos.norm2() + epsilon );
	idOther++;
      }
    }
    energy *= G;
    energy += 0.5*vel.norm2();
    energy *= mass;
    
    energyAll[tid] = energy;
    
  }
  else{
    energyAll[tid] = 0;
  }
}
    
    
    
}//Extern C end
  