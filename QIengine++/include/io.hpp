#pragma once
#include <iostream>
#include <complex>
#include <vector>

#ifndef NDEBUG
    #define DEBUG_CALL(x) x
#else
    #define DEBUG_CALL(x)
#endif

#ifdef CUDA
#define HANDLE_CUDACALL(fzcall) \
    if(fzcall!=cudaSuccess)  \
        printf("ERROR: in %s:%d, call %s, errno %u: %s\n",__FILE__, __LINE__, #fzcall , fzcall, cudaGetErrorString(fzcall));
#else
#define HANDLE_CUDACALL(fzcall) 
#endif

#define CHECK_ERROR(fzcall) \
    {int erri;\
     if((erri = (fzcall))) \
        fprintf(stderr,"ERROR no. %d on call %s, %s:%d\n",erri, #fzcall,__FILE__,__LINE__); \
    }



template <typename T>
std::ostream& operator<<(std::ostream& s, const std::complex<T>& c);


template <typename T>
void sparse_print(std::vector<std::complex<T>> v);

void sparse_print(double *v, uint size);

void sparse_print(double *v, double *w, uint size);

template<class T>
void print(std::vector<T> v);
