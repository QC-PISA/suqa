#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <bits/stdc++.h>
#include <unistd.h>
#include <cmath>
#include <cassert>
#include "Rand.hpp"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "io.hpp"
#include "suqa.cuh"
#include "z2_matter_system.cuh"
#include "Rand.hpp"


using namespace std;

#define NUM_THREADS 128
#define MAXBLOCKS 65535
uint suqa::threads;
uint suqa::blocks;
cudaStream_t suqa::stream1, suqa::stream2;

void deallocate_state(ComplexVec& state){
    if(state.data!=nullptr){
        HANDLE_CUDACALL(cudaFree(state.data));
    }
    state.vecsize=0U;
}

void allocate_state(ComplexVec& state, uint Dim){
    if(state.data!=nullptr or Dim!=state.vecsize)
        deallocate_state(state);


    state.vecsize = Dim; 
    HANDLE_CUDACALL(cudaMalloc((void**)&(state.data), 2*state.vecsize*sizeof(double)));
    // allocate both using re as offset, and im as access pointer.
    state.data_re = state.data;
    state.data_im = state.data_re + state.vecsize;
}


int main(int argc, char** argv){
    if(argc<5){
       printf("usage: %s <m_mass> <total_steps> <trotter_stepsize> <outfile>\n",argv[0]); 
       exit(1);
    }
    m_mass = stod(argv[1]); // (extern) def in src/system.cu
    int total_steps = atoi(argv[2]);
    double trotter_stepsize = stod(argv[3]);
    string outfilename = argv[4];

    printf("arguments:\n m_mass = %.16lg\n total_steps = %d\n trotter_stepsize = %.16lg\n outfile = %s\n", m_mass, total_steps, trotter_stepsize, outfilename.c_str());

    uint Dim = 1U << 7;
    suqa::threads = NUM_THREADS;
    suqa::blocks = (Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>MAXBLOCKS) suqa::blocks=MAXBLOCKS;
    printf("blocks: %u, threads: %u\n",suqa::blocks, suqa::threads);

    ComplexVec state;

    allocate_state(state, Dim);

    pcg rangen;
    rangen.set_seed(time(NULL));

    suqa::setup(Dim);
	 
	init_state(state, Dim);

    DEBUG_CALL(printf("After init:\n"));
    DEBUG_READ_STATE(state);	

	suqa::apply_y(state, bm_z2_qlink0);

    FILE * outfile;
	
    DEBUG_CALL(printf("After Y:\n"));
    DEBUG_READ_STATE(state);	

	suqa::apply_sigmap(state, bm_z2_qlink0);

    DEBUG_CALL(printf("After sigmap:\n"));
    DEBUG_READ_STATE(state);	

	suqa::apply_sigmap(state, bm_z2_qlink0);
 
    DEBUG_CALL(printf("After sigmapsigmap:\n"));
    DEBUG_READ_STATE(state);	

	suqa::apply_sigmam(state, bm_z2_qlink0);
 
    DEBUG_CALL(printf("After sigmapsigmam:\n"));
    DEBUG_READ_STATE(state);	


  
	DEBUG_READ_STATE(state);


return 0;

}
