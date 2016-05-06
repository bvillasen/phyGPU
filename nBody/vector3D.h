#include <iostream>
#include <math.h>
using namespace std;


class Vector3D{
public:
  cudaP x;
  cudaP y;
  cudaP z;
  // Constructor
  __host__ __device__ Vector3D( cudaP x0=0, cudaP y0=0, cudaP z0=0 ) : x(x0), y(y0), z(z0) {}
  // Destructor
//   __host__ __device__ ~Vector3D(){ delete[] &x; delete[] &y; delete[] &z;}
  
  
  __host__ __device__ cudaP norm( void ) { return sqrt( x*x + y*y + z*z ); };
  
  __host__ __device__ cudaP norm2( void ) { return x*x + y*y + z*z; };
  
  __host__ __device__ void normalize(){
    cudaP mag = norm();
    x /= mag;
    y /= mag;
    z /= mag;
  }
  
  __host__ __device__ Vector3D operator+( const Vector3D &v ){
    return Vector3D( x+v.x, y+v.y, z+v.z );
  }
  
  __host__ __device__ Vector3D operator-( const Vector3D &v ){
    return Vector3D( x-v.x, y-v.y, z-v.z );
  }
  
  __host__ __device__ cudaP operator*( const Vector3D &v ){
    return x*v.x + y*v.y + z*v.z;
  }
  
  __host__ __device__ Vector3D operator*( const cudaP a ){
    return Vector3D( a*x, a*y, a*z );
  }  
  
  __host__ __device__ Vector3D& operator+=(const Vector3D &v) {
	  x+=v.x;
	  y+=v.y;
	  z+=v.z;
	  return *this;
  }
  __host__ __device__ Vector3D& operator-=(const Vector3D &v) {
	  x-=v.x;
	  y-=v.y;
	  z-=v.z;
	  return *this;
  }
  __host__ __device__ Vector3D& operator*=(const cudaP a) {
	  x*=a;
	  y*=a;
	  z*=a;
	  return *this;
  }
  
  __host__ __device__ void redefine( cudaP x0, cudaP y0, cudaP z0 ){
    x = x0;
    y = y0;
    z = z0;
  }
  
  
    
  
};
  
