#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include <cassert>
#include "Rand.hpp"
#include <chrono>
#include "io.hpp"
#include "suqa.cuh"
//#include "qms.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

using namespace std;

#define ITERATIONS 100

// externs
uint suqa::threads;
uint suqa::blocks;

#if !defined(CUDA_HOST)

//TODO: maybe, put len and some other constants
//      in device constant memory
__global__ void initialize_state(Complex *state, uint len){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state[i].x =cos((double)(i));
        state[i].y =sin((double)(i*i));
        i += gridDim.x*blockDim.x;
    }
}

#endif



int main(int argc, char** argv){
    if(argc<3){
        printf("usage: %s <qbits> <threads>\n",argv[0]);
        exit(1);
    }


    uint qbits = (uint)atoi(argv[1]);
    uint Dim = (1U<<qbits);
    suqa::threads = (uint)atoi(argv[2]);
    suqa::blocks = (Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>65535) suqa::blocks=65535;

    double vec_norm;

#if defined(CUDA_HOST)
//    uint threads = (Dim<256)? Dim : 256;
    ComplexVec state(Dim);
//    cout<<"Initial vector:"<<endl;
    for(uint i=0; i<state.size(); ++i){
        state[i].x =cos((double)(i));
        state[i].y =sin((double)(i*i));
//        cout<<"i -> ("<<state[i].x<<", "<<state[i].y<<")"<<endl;
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
//    cout<<"\napply vnormalize\n"<<endl;
//    suqa::vnormalize(1,1,state);
//
//    cout<<"Final vector:"<<endl;
//    for(uint i=0; i<state.size(); ++i){
//        cout<<"i -> ("<<state[i].x<<", "<<state[i].y<<")"<<endl;
//    }
//    cout<<"vec_norm = "<<suqa::vnorm(1,1,state)<<endl;
    for(uint j=0; j<ITERATIONS; ++j){
        suqa::apply_h(state, (j)%qbits);
        suqa::apply_x(state, (j+qbits/2)%qbits);
        suqa::apply_cx(state, (j+qbits/3)%qbits, (j+2*qbits/3)%qbits);
        vec_norm = suqa::vnorm(state);
    }
//    vec_norm = suqa::vnorm(1,1,state);

#else
//    Complex* host_state = new Complex[Dim];

    ComplexVec state(Dim);
//    cout<<"Initial vector:"<<Dim<<endl;
    initialize_state<<<suqa::blocks,suqa::threads>>>(state.data,Dim);
    
//    cudaMemcpy(host_state, state.data, Dim*sizeof(Complex), cudaMemcpyDeviceToHost);
//
//    for(uint i=0; i<Dim; ++i){
//        cout<<"i -> ("<<host_state[i].x<<", "<<host_state[i].y<<")"<<endl;
//    }

    cudaDeviceSynchronize();
    auto t_start = std::chrono::high_resolution_clock::now();

    for(uint j=0; j<ITERATIONS; ++j){
        suqa::apply_h(state, (j)%qbits);
        suqa::apply_x(state, (j+qbits/2)%qbits);
        suqa::apply_cx(state, (j+qbits/3)%qbits, (j+2*qbits/3)%qbits);
        vec_norm = suqa::vnorm(state);
    }
//    vec_norm = suqa::vnorm(blocks, threads, state);
//    cout<<"\napply vnormalize\n"<<endl;
//    suqa::vnormalize(state);
//
//    cout<<"Final vector:"<<endl;
//    for(uint i=0; i<state.size(); ++i){
//        cout<<"i -> ("<<state[i].x<<", "<<state[i].y<<")"<<endl;
//    }
//    cout<<"vec_norm = "<<suqa::vnorm(state)<<endl;

//    delete [] host_state;
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs_passed = (1./1000.)*std::chrono::duration<double, std::milli>(t_end-t_start).count();
//	cout<<"All [DONE] in "<<secs_passed<<" seconds"<<endl;

    printf("%u %.4lg %.16lg\n", qbits, secs_passed/ITERATIONS, vec_norm); 
//    printf("%u %.4lg\n", qbits, secs_passed/ITERATIONS); 

    return 0;
}
