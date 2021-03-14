#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <bitset>
#include "io.hpp"
#include "complex_defines.cuh"

#ifdef GPU
#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#endif




#ifndef NDEBUG
#ifdef GPU
extern double *host_state_re, *host_state_im;
#define DEBUG_READ_STATE() {\
    HANDLE_CUDACALL(cudaDeviceSynchronize()); \
    HANDLE_CUDACALL(cudaMemcpyAsync(host_state_re,suqa::state.data_re,suqa::state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream1)); \
    HANDLE_CUDACALL(cudaMemcpyAsync(host_state_im,suqa::state.data_im,suqa::state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream2)); \
    HANDLE_CUDACALL(cudaDeviceSynchronize()); \
    printf("vnorm = %.12lg\n",suqa::vnorm());\
    sparse_print((double*)host_state_re,(double*)host_state_im, suqa::state.size()); \
} 
#else // not GPU

#ifdef SPARSE
#define DEBUG_READ_STATE() {\
    printf("vnorm = %.12lg\n",suqa::vnorm());\
    sparse_print((double*)suqa::state.data_re,(double*)suqa::state.data_im, suqa::state.size(), suqa::actives); \
}
#else // not SPARSE
#define DEBUG_READ_STATE() {\
    printf("vnorm = %.12lg\n",suqa::vnorm());\
    sparse_print((double*)suqa::state.data_re,(double*)suqa::state.data_im, suqa::state.size()); \
}
#endif // ifdef SPARSE

#endif // ifdef GPU

#else  // with NDEBUG
#define DEBUG_READ_STATE()
#endif // ifndef NDEBUG

//const double TWOSQINV = 1./sqrt(2.);
#define TWOSQINV 0.7071067811865475 

typedef std::vector<uint> bmReg;
#define PAULI_ID 0
#define PAULI_X  1
#define PAULI_Y  2
#define PAULI_Z  3


namespace suqa{

extern ComplexVec state;
#ifdef SPARSE
extern std::vector<uint> actives; // dynamic list of active states
#endif

#ifdef GPU
#define NUM_THREADS 128
#define MAXBLOCKS 65535
extern uint blocks, threads;
extern cudaStream_t stream1, stream2;
#endif

// global control mask:
// it applies every next operation 
// using it as condition (the user should make sure
// to use it only for operations not involving it)
extern uint gc_mask;
extern uint nq;

#ifdef GATECOUNT
struct GateRecord{
    uint ng1=0;  // 1 qubit gate 
    uint ng2=0;  // 2 qubit gate
    GateRecord(uint gi1=0,uint gi2=0) : ng1(gi1), ng2(gi2) {}
    GateRecord(const GateRecord& gr) : ng1(gr.ng1), ng2(gr.ng2) {}
};
struct GateCounter{
public:
    std::vector<GateRecord> gs;

    void increment_g1g2(uint ng1, uint ng2){
        if(active){
            gs.back().ng1+=ng1;
            gs.back().ng2+=ng2;
        }
    }
    void increment_g1g2(const GateRecord& gr){
        if(active){
            gs.back().ng1+=gr.ng1;
            gs.back().ng2+=gr.ng2;
        }
    }

    void get_meanstd(double& mean1, double &err1, double& mean2, double& err2){
       get_meanstd(mean1,err1,mean2,err2,gs); 
    }

    inline void activate(){ new_record(); active=true; }
    inline void deactivate(){ active=false; }

    // In a naive implementation, an n-control Toffoli can be described
    // by 2*n-3 standard 2-control Toffoli gates + 1 CNOT, while
    // a single 2-control Toffoli can be written using 9 1-qubit + 6 CNOT gates
    // Some smarter implementations probably exist, 
    // but there are just lower bounds (e.g., see https://arxiv.org/pdf/0803.2316.pdf)
    // We also assume that both types of control (set and unset) are equally available.
    static GateRecord n_ctrl_toffoli_gates(uint n){
        GateRecord gr;
        if(n==0){
            gr.ng1=1; 
        }else if(n==1){
            gr.ng2=1; 
        }else if(n==2){
            gr.ng1=9; 
            gr.ng2=6; 
        }else{
            gr.ng1=9*(2*n-3); 
            gr.ng2=6*(2*n-3)+1; 
        }
        return gr;
    }

    void new_record(){
        gs.push_back(GateRecord());
    }
private:

    // assuming independent samplings
    void get_meanstd(double& mean1, double &err1, double& mean2, double &err2, const std::vector<GateRecord>& gs){
        mean1 = 0.0;
        mean2 = 0.0;
        err1 = 0.0;
        err2 = 0.0;
        for(const auto& el : gs){
            mean1 +=el.ng1;
            mean2 +=el.ng2;
            err1 +=el.ng1*el.ng1;
            err2 +=el.ng2*el.ng2;
        }
        if(gs.size()>0){
            mean1 /= (double)gs.size();
            mean2 /= (double)gs.size();
        }
        if(gs.size()>1){
            err1 = sqrt((err1/(double)gs.size() - mean1*mean1)/(gs.size()-1.0));
            err2 = sqrt((err2/(double)gs.size() - mean2*mean2)/(gs.size()-1.0));
        }
    }

    bool active=false; // unactive by default
};

struct GateCounterList{
    uint gc_mask_set_qbits=0;
    std::vector<GateCounter*> counters;

    void add_counter(GateCounter* gctr){
        counters.push_back(gctr);
    }

    void increment_g1g2(uint ng1, uint ng2){
        for(GateCounter* gc : counters){
            gc->increment_g1g2(ng1,ng2);
        }
    }

    void increment_g1g2(const GateRecord& gr){
        for(GateCounter* gc : counters){
            gc->increment_g1g2(gr.ng1,gr.ng2);
        }
    }

    void update_gc_mask_setbits(){
        uint count=0, n=suqa::gc_mask;
        while(n!=0){
            if((n & 1) == 1)
                count++;
            n>>=1;
        }
        gc_mask_set_qbits=count;
    }

};
extern GateCounterList gatecounters;
#endif


void print_banner();

void activate_gc_mask(const bmReg& q_controls);
void deactivate_gc_mask(const bmReg& q_controls);

/* Utility procedures */
double vnorm();
void vnormalize();

void init_state();

void deallocate_state();
void allocate_state(uint Dim);

/* SUQA gates */
//

// single qbit gates
void apply_x(uint q);
void apply_x(const bmReg& qs);

void apply_y(uint q);
void apply_y(const bmReg& qs);

void apply_z(uint q);
void apply_z(const bmReg& qs);

#ifdef GPU
void apply_sigma_plus(uint q);
void apply_sigma_plus(const bmReg& qs);

void apply_sigma_minus(uint q);
void apply_sigma_minus(const bmReg& qs);
#endif

void apply_h(uint q);
void apply_h(const bmReg& qs);

void apply_t(uint q);
void apply_t(const bmReg& qs);

void apply_tdg(uint q);
void apply_tdg(const bmReg& qs);

void apply_s(uint q);
void apply_s(const bmReg& qs);

// matrix:   1     0
//           0     exp(i phase)
void apply_u1(uint q, double phase);
void apply_u1(uint q, uint q_mask, double phase);

// multiple qbit gates
void apply_cx(const uint& q_control, const uint& q_target, const uint& q_mask=1U);

void apply_mcx(const bmReg& q_controls, const uint& q_target);
void apply_mcx(const bmReg& q_controls, const bmReg& q_mask, const uint& q_target);

void apply_cu1(uint q_control, uint q_target, double phase, uint q_mask=1U);

void apply_mcu1(const bmReg& q_controls, const uint& q_target, double phase);
void apply_mcu1(const bmReg& q_controls, const bmReg& q_mask, const uint& q_target, double phase);

void apply_swap(const uint& q1, const uint& q2);

// apply a list of 2^'q_size' phases, specified in 'phases' to all the combination of qubit states starting from qubit q0 to qubit q0+q_size in the computational basis and standard ordering
void apply_phase_list(uint q0, uint q_size, const std::vector<double>& phases);

// rotation by phase in the direction of a pauli tensor product
void apply_pauli_TP_rotation(const bmReg& q_apply, const std::vector<uint>& pauli_TPconst, double phase);

/* SUQA utils */
void measure_qbit(uint q, uint& c, double rdoub);
void measure_qbits(const bmReg& qs, std::vector<uint>& cs,const std::vector<double>& rdoubs);

void apply_reset(uint q, double rdoub);
void apply_reset(const bmReg& qs, std::vector<double> rdoubs);

void setup(uint nq);
void clear();

void prob_filter(const bmReg& qs, const std::vector<uint>& q_mask, double &prob);

};


