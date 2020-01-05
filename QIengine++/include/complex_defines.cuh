#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cuComplex.h>


typedef cuDoubleComplex Complex; 
struct ComplexVec{
    Complex* data;
    uint vecsize;
    
    ComplexVec(uint vvecsize) : vecsize(vvecsize) {
#if defined(CUDA_HOST)
    //TODO: make allocations using cuda procedures
        data = new Complex[vecsize];
#else
    int err_code = cudaMalloc((void**)&data, vecsize*sizeof(Complex));
    if(err_code!=cudaSuccess)
        printf("ERROR: cudaMalloc errno=%d\n",err_code);
#endif
    }

    ~ComplexVec() {
        
#if defined(CUDA_HOST)
        delete [] data;
#else
        int err_code = cudaFree(data);
        if(err_code!=cudaSuccess)
            printf("ERROR: cudaFree errno=%d\n",err_code);
#endif
    
    }

    size_t size() const{ return vecsize; }

    Complex& operator[](size_t i) { return data[i]; }
    const Complex& operator[](size_t i) const { return data[i]; }


};

const Complex cmplx_one = make_cuDoubleComplex(1.0f, 0.0f);



__host__ __device__ static __inline__ double norm(const Complex& val){
    return val.x*val.x+val.y*val.y;
}


__host__ __device__ static __inline__ Complex& operator/=(Complex& el, const double& fact){
    el.x/=fact;
    el.y/=fact;
    return el;
}



__host__ __device__ static __inline__ Complex& operator+=(Complex& el, const Complex& incr){
    el.x+=incr.x;
    el.y+=incr.y;
    return el;
}
