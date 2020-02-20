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
#include "parser.hpp"
#include "suqa.cuh"
#include "system.cuh"
#include "qms.cuh"

using namespace std;

void print_banner(){
    printf("\n"
"                                          \n" 
"    ███████╗██╗   ██╗ ██████╗  █████╗     \n" 
"    ██╔════╝██║   ██║██╔═══██╗██╔══██╗    \n" 
"    ███████╗██║   ██║██║   ██║███████║    \n" 
"    ╚════██║██║   ██║██║▄▄ ██║██╔══██║    \n" 
"    ███████║╚██████╔╝╚██████╔╝██║  ██║    \n" 
"    ╚══════╝ ╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═╝    \n" 
"                                          \n" 
"\nSimulator for Universal Quantum Algorithms\n\n");
}

#define NUM_THREADS 128
#define MAXBLOCKS 65535
uint suqa::threads;
uint suqa::blocks;
cudaStream_t suqa::stream1, suqa::stream2;


// simulation parameters
double beta;
double h;
int thermalization;

// defined in src/system.cpp
void init_state(ComplexVec& state, uint Dim);

arg_list args;

void save_measures(string outfilename){
    FILE * fil = fopen(outfilename.c_str(), "a");
    for(uint ei = 0; ei < qms::E_measures.size(); ++ei){
        if(qms::Xmatstem!=""){
            fprintf(fil, "%.16lg %.16lg\n", qms::E_measures[ei], qms::X_measures[ei]);
        }else{
            fprintf(fil, "%.16lg\n", qms::E_measures[ei]);
        }
    }
    fclose(fil);
    qms::E_measures.clear();
    qms::X_measures.clear();
}

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
    if(argc < 8){
        printf("usage: %s <beta> <g_beta> <metro steps> <reset each> <num state qbits> <num ene qbits> <output file path> [--max-reverse <max reverse attempts>=20] [--seed <seed>=random] [--PE-time <factor for time in PE (coeff. of 2pi)>=1.0] [--PE-steps <steps of PE evolution>=10] [--thermalization <steps>=100] [--X-mat-stem <stem for X measure matrix>] [--record-reverse]\n", argv[0]);
        exit(1);
    }

    parse_arguments(args, argc, argv);

    beta = args.beta;
    g_beta = args.g_beta; // defined as extern in system.cuh
    thermalization = args.thermalization;
    qms::metro_steps = (uint)args.metro_steps;
    qms::reset_each = (uint)args.reset_each;
    qms::state_qbits = (uint)args.state_qbits;
    qms::ene_qbits = (uint)args.ene_qbits;
    string outfilename(args.outfile);
    qms::max_reverse_attempts = (uint)args.max_reverse_attempts;
    qms::t_PE_factor = args.pe_time_factor;
    qms::t_phase_estimation = qms::t_PE_factor*8.*atan(1.0); // 2*pi*t_PE_factor
    qms::n_phase_estimation = args.pe_steps;
    qms::Xmatstem = args.Xmatstem;
    qms::record_reverse= args.record_reverse;
    qms::iseed = args.seed;
    if(qms::iseed>0)
        qms::rangen.set_seed(qms::iseed);
    
    qms::iseed = qms::rangen.get_seed();

    qms::nqubits = qms::state_qbits + 2*qms::ene_qbits + 1;
    qms::Dim = (1U << qms::nqubits);
    qms::ene_levels = (1U << qms::ene_qbits);
    qms::state_levels = (1U << qms::state_qbits);

    suqa::threads = NUM_THREADS;
    suqa::blocks = (qms::Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>MAXBLOCKS) suqa::blocks=MAXBLOCKS;

    
    // Banner
    print_banner();
    cout<<"arguments:\n"<<args<<endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Initialization of utilities
    suqa::setup(qms::Dim);
    qms::setup(beta);

    // Initialization:
    // known eigenstate of the system (see src/system.cu)
    
    allocate_state(qms::gState, qms::Dim);
    init_state(qms::gState, qms::Dim);

    //TODO: make it an args option?
    uint perc_mstep = (qms::metro_steps+19)/20; // batched saves
    
    if( access( outfilename.c_str(), F_OK ) == -1 ){
        FILE * fil = fopen(outfilename.c_str(), "w");
        fprintf(fil, "# E%s\n",(qms::Xmatstem!="")?" A":"");
        fclose(fil);
    }

    bool take_measure;
    uint s0 = 0U;
    for(uint s = 0U; s < qms::metro_steps; ++s){
        DEBUG_CALL(cout<<"metro step: "<<s<<endl);
        take_measure = (s>s0+(uint)thermalization and (s-s0)%qms::reset_each ==0U);
        int ret = qms::metro_step(take_measure);

        if(ret<0){ // failed rethermalization, reinitialize state
            init_state(qms::gState, qms::Dim);
            //ensure new rethermalization
            s0 = s+1; 
        }
        if(s%perc_mstep==0){
            cout<<"iteration: "<<s<<"/"<<qms::metro_steps<<endl;
            save_measures(outfilename);
        }
    }

    cout<<endl;
    deallocate_state(qms::gState);
    qms::clear();
    suqa::clear();

    cout<<"\nall fine :)\n"<<endl;



    if(qms::record_reverse){
        FILE * fil_rev = fopen((outfilename+"_revcounts").c_str(), "w");


        for(uint i = 0; i < qms::reverse_counters.size(); ++i){
            fprintf(fil_rev, "%d %d\n", i, (int)qms::reverse_counters[i]);
        }
        fclose(fil_rev);
    }

    cout<<"\n\tSuqa!\n"<<endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs_passed = (1./1000.)*std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cout<<"All [DONE] in "<<secs_passed<<" seconds"<<endl;

    return 0;
}
