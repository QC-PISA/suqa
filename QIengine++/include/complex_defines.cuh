#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cuComplex.h>


typedef cuDoubleComplex Complex; 
struct ComplexVec{
    uint vecsize;
    Complex* data=nullptr;
    
    ComplexVec() : vecsize(0U), data(nullptr) {}

    ComplexVec(uint vvecsize) {
//        // allocation and deallocation 
//        // are managed by external methods called in main()
//        // to prevent cudaErrorCudartUnloading
////        allocate(vvecsize);
#if defined(CUDA_HOST)
    //TODO: make allocations using cuda procedures
        vecsize = vvecsize;
        data = new Complex[vecsize];
#else
        printf("WARNING: allocations and deallocations" 
               "are managed by external methods called in main()"
               "to prevent cudaErrorCudartUnloading");
#endif
    }

    ~ComplexVec() {
        // allocation and deallocation 
        // are managed by external methods called in main()
        // to prevent cudaErrorCudartUnloading
//        deallocate();        
#if defined(CUDA_HOST)
        if(data!=nullptr)
            delete [] data;
        data = nullptr;
        vecsize = 0U;
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

__host__ __device__ static __inline__ Complex& operator*=(Complex& el, const Complex& fact){
    double tmpval = el.x;
    el.x = el.x*fact.x - el.y*fact.y;
    el.y = tmpval*fact.y + el.y*fact.x;
    return el;
}

__host__ __device__ static __inline__ Complex operator+(const Complex& a, const Complex& b){
    Complex ret;
    ret.x = a.x+b.x;
    ret.y = a.y+b.y;
    return ret;
}

__host__ __device__ static __inline__ Complex operator*(const Complex& el, const Complex& fact){
    Complex ret;
    ret.x = el.x*fact.x - el.y*fact.y;
    ret.y = el.x*fact.y + el.y*fact.x;
    return ret;
}

__host__ __device__ static __inline__ Complex operator*(const Complex& el, const double& fact){
    Complex ret;
    ret.x = el.x*fact;
    ret.y = el.y*fact;
    return ret;
}

__host__ __device__ static __inline__ Complex operator*(const double& fact, const Complex& el){
    Complex ret;
    ret.x = el.x*fact;
    ret.y = el.y*fact;
    return ret;
}


// this is expi(z) == exp(i z)
__host__ __device__ static __inline__ Complex expi(double z){

    Complex res;

    sincos (z, &res.y, &res.x);

    return res;

}

