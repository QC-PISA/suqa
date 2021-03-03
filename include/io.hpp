#pragma once
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <bitset>
#include <chrono>

#define ARED     "\x1b[31m"
#define AGREEN   "\x1b[32m"
#define AYELLOW  "\x1b[33m"
#define ABLUE    "\x1b[34m"
#define AMAGENTA "\x1b[35m"
#define ACYAN    "\x1b[36m"
#define ARESET   "\x1b[0m"

typedef unsigned int uint;
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

int get_time(struct timeval* tp, struct timezone* tzp);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__) && !defined(__MINGW64__)
#define and &&
#define not !
#define or ||
#pragma comment(lib, "Shlwapi.lib")
#include <Shlwapi.h>
#include <winsock.h>
#else
#include <unistd.h>
#endif


#ifndef NDEBUG
    #define DEBUG_CALL(x) x
#else
    #define DEBUG_CALL(x)
#endif

//#ifdef CUDA
#define HANDLE_CUDACALL(fzcall) \
    if((fzcall)!=cudaSuccess)  \
        printf("ERROR: in %s:%d, call %s, errno %u: %s\n",__FILE__, __LINE__, #fzcall , fzcall, cudaGetErrorString(fzcall));
//#else
//#define HANDLE_CUDACALL(fzcall) 
//#endif

#define CHECK_ERROR(fzcall) \
    {int erri;\
     if((erri = (fzcall))) \
        fprintf(stderr,"ERROR no. %d on call %s, %s:%d\n",erri, #fzcall,__FILE__,__LINE__); \
    }


bool file_exists(const char* fname);

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::complex<T>& c);

// sparse_print() prints arrays with non-zero entries (threshold 1e-10; see src)
template <typename T>
void sparse_print(std::vector<std::complex<T>> v);

void sparse_print(double *v, uint size);

void sparse_print(double *v, double *w, uint size);

void qoxo_print(double *v, double *w, uint size);

template<class T>
void print(std::vector<T> v);

