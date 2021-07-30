#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
//#include <bits/stdc++.h>
//#include <unistd.h>
#include <cmath>
#include <cassert>
#include "Rand.hpp"
#include <chrono>
#ifdef GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif
#include "io.hpp"
#include "parser_qms.hpp"
#include "suqa.cuh"
#include "system.cuh"
#include "qms.cuh"

using namespace std;



// simulation parameters
double beta;
double h;
int thermalization;

// defined in src/system.cu
void init_state();

arg_list args;

#ifdef GATECOUNT
GateCounter gctr_global("global");
GateCounter gctr_metrostep("metro step");
GateCounter gctr_sample("sample");
GateCounter gctr_measure("measure");
GateCounter gctr_reverse("reverse");
#endif

void save_measures(string outfilename){
    FILE * fil = fopen(outfilename.c_str(), "a");
    for(uint ei = 0; ei < qms::E_measures.size(); ++ei){
        fprintf(fil, "%.16lg %.16lg\n", qms::E_measures[ei], qms::X_measures[ei]);
    }
    fclose(fil);
    qms::E_measures.clear();
    qms::X_measures.clear();
}

int main(int argc, char** argv){
    if(argc < 7){
        printf("usage: %s <beta> <metro steps> <reset each> <num syst qbits> <num ene qbits> <output file path> [--max-reverse <max reverse attempts> (20)] [--seed <seed> (random)] [--ene-min <min energy> (0.0)] [--ene-max <max energy> (1.0)] [--PE-steps <steps of PE evolution> (10)] [--thermalization <steps> (100)] [--record-reverse]\n", argv[0]);
        exit(1);
    }

    parse_arguments(args, argc, argv);

    beta = args.beta;
//    g_beta = args.g_beta; // defined as extern in system.cuh
    thermalization = args.thermalization;
    qms::metro_steps = (uint)args.metro_steps;
    qms::reset_each = (uint)args.reset_each;
    qms::syst_qbits = (uint)args.syst_qbits;
    qms::ene_qbits = (uint)args.ene_qbits;
    string outfilename(args.outfile);
    qms::max_reverse_attempts = (uint)args.max_reverse_attempts;
    qms::n_phase_estimation = args.pe_steps;
    qms::record_reverse= args.record_reverse;
    qms::iseed = args.seed;
    if(qms::iseed>0)
        qms::rangen.set_seed(qms::iseed);
    
    qms::iseed = qms::rangen.get_seed();

    qms::nqubits = qms::syst_qbits + 2*qms::ene_qbits + 1;
    qms::Dim = (1U << qms::nqubits);
    qms::ene_levels = (1U << qms::ene_qbits);
    qms::syst_levels = (1U << qms::syst_qbits);

    qms::t_PE_shift = args.ene_min;
    qms::t_PE_factor = (qms::ene_levels-1)/(double)(qms::ene_levels*(args.ene_max-args.ene_min)); 
    qms::t_phase_estimation = qms::t_PE_factor*8.*atan(1.0); // 2*pi*t_PE_factor

    
    // Banner
    suqa::print_banner();
    cout<<"arguments:\n"<<args<<endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Initialization of utilities
    suqa::setup(qms::nqubits);
    qms::setup(beta);

#ifdef GATECOUNT
    suqa::gatecounters.add_counter(&gctr_global);
    suqa::gatecounters.add_counter(&gctr_metrostep);
    suqa::gatecounters.add_counter(&gctr_sample);
    suqa::gatecounters.add_counter(&gctr_measure);
    suqa::gatecounters.add_counter(&gctr_reverse);

    gctr_global.new_record();
    gctr_sample.new_record();
#endif


    // Initialization:
    // known eigenstate of the system (see src/system.cu)
    
    DEBUG_CALL(cout<<"Preinitial state: "<<endl);
    DEBUG_READ_STATE();
    init_state();
    DEBUG_CALL(cout<<"Initial state: "<<endl);
    DEBUG_READ_STATE();


    //TODO: make it an args option?
    uint perc_mstep = (qms::metro_steps+19)/20; // batched saves
    
    uint count_accepted = 0U;
    if(!file_exists(outfilename.c_str())){
        FILE * fil = fopen(outfilename.c_str(), "w");
        fprintf(fil, "# E A\n");
        fclose(fil);
    }

    bool take_measure;
    uint s0 = 0U;
    //TODO: change metro_steps into actual measures sampled?
    for(uint s = 0U; s < qms::metro_steps; ++s){
        DEBUG_CALL(cout<<"metro step: "<<s<<endl);
        take_measure = (s>s0+(uint)thermalization and (s-s0)%qms::reset_each ==0U);
        int ret = qms::metro_step(take_measure);

        if(ret<0){ // failed rethermalization, reinitialize state
            init_state();
            //ensure new rethermalization
            s0 = s+1; 
        }
        if(ret==1 or ret==2){
            count_accepted++;
        }
        if(s%perc_mstep==0){
            cout<<"iteration: "<<s<<"/"<<qms::metro_steps<<endl;
            save_measures(outfilename);
        }
    }

    cout<<endl;
    printf("\n\tacceptance: %3.2lg%%\n",(count_accepted/static_cast<double>(qms::metro_steps))*100.0);


    qms::clear();
    suqa::clear();

    cout<<"\nall fine :)\n"<<endl;



    if(qms::record_reverse){
        FILE * fil_rev = fopen((outfilename+"_revcounts").c_str(), "w");

        for(uint i = 0; i < qms::reverse_counters.size(); ++i){
            fprintf(fil_rev, "%d %d\n", i, static_cast<int>(qms::reverse_counters[i]));
        }
        fclose(fil_rev);
    }

    cout<<"\n\tSuqa!\n"<<endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs_passed = (1./1000.)*std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cout<<"All [DONE] in "<<secs_passed<<" seconds"<<endl;

    return 0;
}
