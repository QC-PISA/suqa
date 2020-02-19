// -*- C++ -*-
//^ this is a command that sets emacs in c++-mode
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
#include "system.cuh"
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
//definition in src/system.cu
void self_plaquette(ComplexVec& state, const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3); //calculate plaquette writing it on ... state?

int main(int argc, char** argv){
    if(argc<5){
       printf("usage: %s <g_beta> <total_steps> <trotter_stepsize> <outfile>\n",argv[0]); 
       exit(1);
    }
    g_beta = stod(argv[1]); // (extern) def in src/system.cu
    int total_steps = atoi(argv[2]);
    double trotter_stepsize = stod(argv[3]);
    string outfilename = argv[4];

    printf("arguments:\n g_beta = %.16lg\n total_steps = %d\n trotter_stepsize = %.16lg\n outfile = %s\n", g_beta, total_steps, trotter_stepsize, outfilename.c_str());

    uint Dim = 1U << 5;//5 is the number of ideal (qu)bits we need to do the simulation. It may be 4
    suqa::threads = NUM_THREADS;
    suqa::blocks = (Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>MAXBLOCKS) suqa::blocks=MAXBLOCKS;
    printf("blocks: %u, threads: %u\n",suqa::blocks, suqa::threads);
    
    ComplexVec state;
    
    allocate_state(state, Dim);

    // pcg rangen;
    // rangen.set_seed(time(NULL));

    suqa::setup(Dim);
    init_state(state, Dim); //remember to correctly initialize the state

    FILE * outfile;

    DEBUG_CALL(printf("initial state:\n"));
    DEBUG_READ_STATE(state);

    for(uint ii=0; ii<=(uint)total_steps; ++ii){
      double t = ii*trotter_stepsize;
      printf("time %.16lg\n", t);
      double plaq_val=0.0; 
      double plaq_val_std=0.0;
//            for(uint hit=0; hit<(uint)num_hits; ++hit){
//            printf("\thit %u\n", hit);
//            init_state(state, Dim, g_beta);
//            evolution(state, t, ii);
//            self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
//            std::vector<uint> c(3);
//            suqa::measure_qbit(state,bm_qlink1[0],c[0],rangen.doub());
//            suqa::measure_qbit(state,bm_qlink1[1],c[1],rangen.doub());
//            suqa::measure_qbit(state,bm_qlink1[2],c[2],rangen.doub());
//            uint plaq_idx = 4*c[2]+2*c[1]+c[0];
//            double plaq_tmp = (plaq_idx==0)? 2.0 : ((plaq_idx==2)? -2.0 : 0.0);
//            plaq_val += plaq_tmp;
//            plaq_val_std += plaq_tmp*plaq_tmp;
//        }
//        plaq_val /=(double)num_hits;
//        plaq_val_std = sqrt((plaq_val_std/(double)num_hits - plaq_val*plaq_val)/(double)(num_hits-1));
//        fprintf(outfile, "%.16lg %d %.16lg %.16lg\n", t, num_hits, plaq_val, plaq_val_std);

        init_state(state, Dim);
        evolution(state, t, ii);
        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);

        double p0;
        suqa::prob_filter(state, bm_qlink1, {0U}, p0);
        printf("p000 = %.12lg\n", p0);
        plaq_val = 2.0*(p0);
        plaq_val_std = sqrt(4.0*(p0)-plaq_val*plaq_val);
        outfile = fopen(outfilename.c_str(), "a");
        fprintf(outfile, "%.12lg %.12lg %.12lg\n", t, plaq_val, plaq_val_std);
        printf("%.12lg %.12lg %.12lg\n", t, plaq_val, plaq_val_std);
        fclose(outfile);
    }
    

    suqa::clear();
    
    deallocate_state(state);


    return 0;
}
