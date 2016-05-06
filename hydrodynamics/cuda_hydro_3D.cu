#include <pycuda-helpers.hpp>

//Textures for conserv
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_1;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_2;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_3;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_4;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_5;


//Surfaces for Fluxes
surface< void, cudaSurfaceType3D> surf_1;
surface< void, cudaSurfaceType3D> surf_2;
surface< void, cudaSurfaceType3D> surf_3;
surface< void, cudaSurfaceType3D> surf_4;
surface< void, cudaSurfaceType3D> surf_5;

__global__ void setInterFlux_hll( const int coord, const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz,
			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5,
			 float* times ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP v2;
  cudaP rho_l, vx_l, vy_l, vz_l, E_l, p_l;
  cudaP rho_c, vx_c, vy_c, vz_c, E_c, p_c;
//   float time;
  //Read adjacent conserv
  if ( coord == 1 ){
    rho_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);

    vx_l  = fp_tex3D( tex_2, t_j-1, t_i, t_k) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;

    vy_l  = fp_tex3D( tex_3, t_j-1, t_i, t_k) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;

    vz_l  = fp_tex3D( tex_4, t_j-1, t_i, t_k) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;

    E_l   = fp_tex3D( tex_5, t_j-1, t_i, t_k);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);


  }
  else if ( coord == 2 ){
    rho_l = fp_tex3D( tex_1, t_j, t_i-1, t_k);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);

    vx_l  = fp_tex3D( tex_2, t_j, t_i-1, t_k) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;

    vy_l  = fp_tex3D( tex_3, t_j, t_i-1, t_k) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;

    vz_l  = fp_tex3D( tex_4, t_j, t_i-1, t_k) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;

    E_l   = fp_tex3D( tex_5, t_j, t_i-1, t_k);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);


  }
  else if ( coord == 3 ){
    rho_l = fp_tex3D( tex_1, t_j, t_i, t_k-1);
    rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);

    vx_l  = fp_tex3D( tex_2, t_j, t_i, t_k-1) / rho_l;
    vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;

    vy_l  = fp_tex3D( tex_3, t_j, t_i, t_k-1) / rho_l;
    vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;

    vz_l  = fp_tex3D( tex_4, t_j, t_i, t_k-1) / rho_l;
    vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;

    E_l   = fp_tex3D( tex_5, t_j, t_i, t_k-1);
    E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);


  }
//   //Boundary bounce condition
//     if ( t_j == 0 ) vx_l = -vx_c;
//       //Boundary bounce condition
//     if ( t_i == 0 ) vy_l = -vy_c;
//     //Boundary bounce condition
//     if ( t_k == 0 ) vz_l = -vz_c;

  v2    = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l;
  p_l   = ( E_l - rho_l*v2/2 ) * (gamma-1);

  v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
  p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);


  cudaP cs_l, cs_c, s_l, s_c;
  cs_l = sqrt( p_l * gamma / rho_l );
  cs_c = sqrt( p_c * gamma / rho_c );



  if ( coord == 1 ){
    s_l = min( vx_l - cs_l, vx_c - cs_c );
    s_c = max( vx_l + cs_l, vx_c + cs_c );
    //Use v2 to save time minimum
    v2 = dx / ( abs( vx_c ) + cs_c );
    v2 = min( v2, dy / ( abs( vy_c ) + cs_c ) );
    v2 = min( v2, dz / ( abs( vz_c ) + cs_c ) );
    times[ tid ] = v2;
  }
  else if ( coord == 2 ){
    s_l = min( vy_l - cs_l, vy_c - cs_c );
    s_c = max( vy_l + cs_l, vy_c + cs_c );
  }
  else if ( coord == 3 ){
    s_l = min( vz_l - cs_l, vz_c - cs_c );
    s_c = max( vz_l + cs_l, vz_c + cs_c );
  }

  // Adjacent fluxes from left and center cell
  cudaP F_l, F_c, iFlx;

  //iFlx rho
  if ( coord == 1 ){
    F_l = rho_l * vx_l;
    F_c = rho_c * vx_c;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vy_l;
    F_c = rho_c * vy_c;
  }
  else if ( coord == 3 ){
    F_l = rho_l * vz_l;
    F_c = rho_c * vz_c;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c - rho_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);

  //iFlx rho * vx
  if ( coord == 1 ){
    F_l = rho_l * vx_l * vx_l + p_l;
    F_c = rho_c * vx_c * vx_c + p_c;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vx_l * vy_l;
    F_c = rho_c * vx_c * vy_c;
  }
  else if ( coord == 3 ){
    F_l = rho_l * vx_l * vz_l;
    F_c = rho_c * vx_c * vz_c;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vx_c - rho_l*vx_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_2,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);

  //iFlx rho * vy
  if ( coord == 1 ){
    F_l = rho_l * vy_l * vx_l ;
    F_c = rho_c * vy_c * vx_c ;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vy_l * vy_l + p_l;
    F_c = rho_c * vy_c * vy_c + p_c;
  }
  else if ( coord == 3 ){
    F_l = rho_l * vy_l * vz_l;
    F_c = rho_c * vy_c * vz_c;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vy_c - rho_l*vy_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_3,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);

  //iFlx rho * vz
  if ( coord == 1 ){
    F_l = rho_l * vz_l * vx_l ;
    F_c = rho_c * vz_c * vx_c ;
  }
  else if ( coord == 2 ){
    F_l = rho_l * vz_l * vy_l ;
    F_c = rho_c * vz_c * vy_c ;
  }
  else if ( coord == 3 ){
    F_l = rho_l * vz_l * vz_l + p_l ;
    F_c = rho_c * vz_c * vz_c + p_c ;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vz_c - rho_l*vz_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_4,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);

  //iFlx E
  if ( coord == 1 ){
    F_l = vx_l * ( E_l + p_l ) ;
    F_c = vx_c * ( E_c + p_c ) ;
  }
  else if ( coord == 2 ){
    F_l = vy_l * ( E_l + p_l ) ;
    F_c = vy_c * ( E_c + p_c ) ;
  }
  else if ( coord == 3 ){
    F_l = vz_l * ( E_l + p_l ) ;
    F_c = vz_c * ( E_c + p_c ) ;
  }
  if ( s_l > 0 ) iFlx = F_l;
  else if ( s_c < 0 ) iFlx = F_c;
  else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( E_c - E_l ) ) / ( s_c - s_l );
  surf3Dwrite(  iFlx, surf_5,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);

}

__global__ void getInterFlux_hll( const int coord, const cudaP dt,  const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz,
			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5,
			 cudaP* gForceX, cudaP* gForceY, cudaP* gForceZ, cudaP* gravWork ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  //Read inter-cell fluxes from textures

  cudaP iFlx1_l, iFlx2_l, iFlx3_l, iFlx4_l, iFlx5_l;
  cudaP iFlx1_r, iFlx2_r, iFlx3_r, iFlx4_r, iFlx5_r;
  cudaP delta;
  if ( coord == 1 ){
    delta = dt / dx;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);

    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j+1, t_i, t_k);

    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j+1, t_i, t_k);

    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j+1, t_i, t_k);

    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j+1, t_i, t_k);
  }
  else if ( coord == 2 ){
    delta = dt / dy;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j, t_i+1, t_k);

    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j, t_i+1, t_k);

    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j, t_i+1, t_k);

    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j, t_i+1, t_k);

    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j, t_i+1, t_k);
  }
  else if ( coord == 3 ){
    delta = dt / dz;
    iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
    iFlx1_r = fp_tex3D( tex_1, t_j, t_i, t_k+1);

    iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
    iFlx2_r = fp_tex3D( tex_2, t_j, t_i, t_k+1);

    iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
    iFlx3_r = fp_tex3D( tex_3, t_j, t_i, t_k+1);

    iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
    iFlx4_r = fp_tex3D( tex_4, t_j, t_i, t_k+1);

    iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
    iFlx5_r = fp_tex3D( tex_5, t_j, t_i, t_k+1);
  }

  //Advance the consv values
  // cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
  // cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l ) + dt*gForceX[tid]*50;
  // cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l ) + dt*gForceY[tid]*50;
  // cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l ) + dt*gForceZ[tid]*50;
  // cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l ) + dt*gravWork[tid]*50;

  cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
  cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l );
  cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l );
  cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l );
  cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l );
}


__global__ void iterPoissonStep( int* converged, const int paridad,
				 const int nWidth, const cudaP omega, const cudaP pi4,
				 cudaP dx, cudaP dy, cudaP dz,
				 cudaP* rhoVals, cudaP* phiVals, float* phiWall ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  //Make a checkboard 3D grid
  if ( t_i%2 == 0 ){
    if ( t_k%2 == paridad ) t_j +=1;
  }
  else if ( (t_k+1)%2 == paridad ) t_j +=1;
  int tid = t_j + t_i*nWidth + t_k*nWidth*blockDim.y*gridDim.y;

  cudaP rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
  rho = rhoVals[ tid ];
  phi_c = fp_tex3D( tex_1, t_j, t_i, t_k);
  phi_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
  phi_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
  phi_d = fp_tex3D( tex_1, t_j, t_i-1, t_k);
  phi_u = fp_tex3D( tex_1, t_j, t_i+1, t_k);
  phi_b = fp_tex3D( tex_1, t_j, t_i, t_k-1);
  phi_t = fp_tex3D( tex_1, t_j, t_i, t_k+1);

  //Boundary conditions
  if  ( t_j == 0 )        phi_l = phi_r;
  if  ( t_j == nWidth-1 ) phi_r = phi_l;
  if  ( t_i == 0 )        phi_d = phi_u;
  if  ( t_i == nWidth-1 ) phi_u = phi_d;
  if  ( t_k == 0 )        phi_b = phi_t;
  if  ( t_k == nWidth-1 ) phi_t = phi_b;

//   phi_new =  1./6 * ( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*rho   );
  phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*pi4*rho );

  if ( paridad == 0 ) surf3Dwrite(  phi_new, surf_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
  phiVals[ tid ] = phi_new;

//   if ( ( t_j == 0 ) ||  ( t_j == nWidth-1 ) || ( t_i == 0 ) || ( t_i == nWidth-1 ) || ( t_k == 0 ) || ( t_k == nWidth-1 ) ) return;
//   if ( ( blockIdx.x == 0 ) ||  ( blo == nWidth-1 ) || ( t_i == 0 ) || ( t_i == nWidth-1 ) || ( t_k == 0 ) || ( t_k == nWidth-1 ) ) return;

  if ( ( abs( ( phi_new - phi_c ) / phi_c ) > 0.002 ) ) converged[0] = 0;


}

__global__ void getGravityForce( const int nWidth, const int nHeight, const int nDepth,
				 cudaP dx, cudaP dy, cudaP dz,
				 cudaP* gForce_x, cudaP* gForce_y, cudaP* gForce_z,
				 cudaP* rho, cudaP* pX, cudaP* pY, cudaP* pZ, cudaP *gravWork,
				 float* phiWall      ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP phi_l, phi_r, phi_d, phi_u, phi_b, phi_t;
//   phi_c = fp_tex3D( tex_1, t_j, t_i, t_k);
  phi_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
  phi_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
  phi_d = fp_tex3D( tex_1, t_j, t_i-1, t_k);
  phi_u = fp_tex3D( tex_1, t_j, t_i+1, t_k);
  phi_b = fp_tex3D( tex_1, t_j, t_i, t_k-1);
  phi_t = fp_tex3D( tex_1, t_j, t_i, t_k+1);

  //Boundary conditions
  if  ( t_j == 0 )        phi_l = phi_r;
  if  ( t_j == nWidth-1 ) phi_r = phi_l;
  if  ( t_i == 0 )        phi_d = phi_u;
  if  ( t_i == nWidth-1 ) phi_u = phi_d;
  if  ( t_k == 0 )        phi_b = phi_t;
  if  ( t_k == nWidth-1 ) phi_t = phi_b;

  //Get partial derivatives for force
  cudaP gField_x, gField_y, gField_z, p_x, p_y, p_z, rho_c;
  rho_c = rho[ tid ];
  gField_x = ( phi_l - phi_r ) * 0.5 / dx;
  gField_y = ( phi_d - phi_u ) * 0.5 / dy;
  gField_z = ( phi_b - phi_t ) * 0.5 / dz;
  gForce_x[ tid ] = gField_x * rho_c;
  gForce_y[ tid ] = gField_y * rho_c;
  gForce_z[ tid ] = gField_z * rho_c;
//   gForce_x[ tid ] = gField_x;
//   gForce_y[ tid ] = gField_y;
//   gForce_z[ tid ] = gField_z;

  //Get momentum for virtual gravitational work
  p_x = pX[ tid ] ;
  p_y = pY[ tid ] ;
  p_z = pZ[ tid ] ;
  gravWork[ tid ] = p_x * gField_x + p_y * gField_y + p_z * gField_z ;

}

__global__ void reduceDensity( const int nWidth, const int nHeight, const int nDepth,
			       const float dx, const float dy, const float dz,
			       const float xMin, const float yMin, const float zMin,
			       cudaP* rhoAll, float* rhoReduced,
			       float* blockX, float* blockY, float* blockZ  ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

  float rho = float( rhoAll[tid] );
  __shared__ float density[ THREADS_PER_BLOCK ];

  density[tid_b] = rho;
  __syncthreads();

  int i =  blockDim.x * blockDim.y * blockDim.z  / 2;
  while ( i > 0 ){
    if ( tid_b < i ) density[tid_b] = density[tid_b] + density[tid_b+i];
    __syncthreads();
    i /= 2;
  }


  float x = blockDim.x*dx * ( blockIdx.x + 0.5f ) + xMin;
  float y = blockDim.y*dy * ( blockIdx.y + 0.5f ) + yMin;
  float z = blockDim.z*dz * ( blockIdx.z + 0.5f ) + zMin;
  if (tid_b == 0 ){
    rhoReduced[ bid ] = density[0]*dx*dy*dz ;
    blockX[ bid ] = x;
    blockY[ bid ] = y;
    blockZ[ bid ] = z;
  }

}
__global__ void getBounderyPotential(const float pi4, const int nBlocks, const int nWidth, const int nHeight, const int nDepth,
			      float dx, float dy, float dz, float xMin, float yMin, float zMin,
			      float* rhoReduced,  float* phiWall,
			      float* blockX, float* blockY, float* blockZ   ){
// 			      float* phiWall_l, float* phiWall_r, float* phiWall_d, float* phiWall_u, float* phiWall_b, float* phiWall_t){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_j + t_i*blockDim.x*gridDim.x ;
//   int tid_b = threadIdx.x + threadIdx.y*blockDim.x
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

  float y_wall = yMin + t_j*dy;
  float z_wall = zMin + t_i*dz;

  float x_b, y_b, z_b, phi, rho ;
  phi = 0;

  for ( int nBlock = 0; nBlock<nBlocks; nBlock++ ){
    rho = rhoReduced[ nBlock ];
    x_b = blockX[nBlock];
    y_b = blockY[nBlock];
    z_b = blockZ[nBlock];
    phi -= rsqrt( x_b*x_b + (y_b-y_wall)*(y_b-y_wall) + (z_b-z_wall)*(z_b-z_wall) ) * rho;
  }
  phiWall[ tid ] = phi;

}



// __global__ void getBounderyPotential( const int nWidth, const int nHeight, const int nDepth,
// 			      float dx, float dy, float dz, float xMin, float yMin, float zMin,
// 			      cudaP* rhoAll, float* phiWall ){
// // 			      float* phiWall_l, float* phiWall_r, float* phiWall_d, float* phiWall_u, float* phiWall_b, float* phiWall_t){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//
//   const int ny = 8*8*2;
//   const int nz = 2;
//
//   const int nSwipes = 50;
//
//   float rho = float ( rhoAll[ tid ] );
//
//   float x, y, z, y_wall, z_wall, phi;
//   x = xMin + t_j*dx;
//   y = yMin + t_i*dy;
//   z = zMin + t_k*dz;
//
//   int idx_1, idx_2, i, j, swipeCounter, swipeIdx;
//   swipeIdx = tid / 256;
//
//   //Allocate shared memory
//   __shared__ float wallStripe[ 256 ];
//
//   for ( swipeCounter=0; swipeCounter<nSwipes; swipeCounter++ ){
//     //Initialize shared memory to zero
//     wallStripe[ tid_b ] = 0;
//     __syncthreads();
//
//     //Initialize the indexes over the tile
//     idx_1 = tid_b % ny;
//     idx_2 = tid_b / ny;
//
//     //Fill the tile of the wall
//     for ( j=0; j<nz; j++ ){
//       z_wall = idx_2*dz + zMin;
//       for ( i=0; i<ny; i++ ){
// 	y_wall = idx_1*dy + yMin;
// 	phi = rsqrt( x*x + (y-y_wall)*(y_wall) + (z-z_wall)*(z-z_wall) ) * rho;
// 	wallStripe[ idx_2*ny + idx_1 ] += phi;
// 	idx_1 += 1;
// 	if ( idx_1 >= ny ) idx_1 = 0;
//       }
//       idx_2 += 1;
//       if ( idx_2 >= nz ) idx_2 = 0;
//     }
//
//     //Write the tile values to global memory
//     idx_1 = tid_b % ny;
//     idx_2 = tid_b / ny;
//     atomicAdd( &phiWall[ swipeIdx*256 + idx_2*ny + idx_1  ], wallStripe[ tid_b ] ) ;
// //     swipeIdx += 1;
// //     if ( swipeIdx >= 128 ) swipeIdx = 0;
//   }
//
// }
