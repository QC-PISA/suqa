#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <bitset>
#include "io.hpp"
#include "complex_defines.cuh"
#include "gatecounters.hpp"

#ifdef GPU
#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
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

extern std::vector<double> rphase_m; // for the QFT

#ifdef GATECOUNT
extern GateCounterList gatecounters;
#endif


void print_banner();

void activate_gc_mask(const bmReg& q_controls);
void deactivate_gc_mask(const bmReg& q_controls);

/* Utility procedures */
double vnorm();
void vnormalize();

void init_state();
void init_state(std::vector<double> re_coeff, std::vector<double> im_coeff);

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

//void apply_sigma_plus(uint q);
//void apply_sigma_plus(const bmReg& qs);
//
//void apply_sigma_minus(uint q);
//void apply_sigma_minus(const bmReg& qs);

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

void apply_qft(const std::vector<uint>& qact);
void apply_qft_inverse(const std::vector<uint>& qact);

/* SUQA utils */
void measure_qbit(uint q, uint& c, double rdoub);
void measure_qbits(const bmReg& qs, std::vector<uint>& cs,const std::vector<double>& rdoubs);

void apply_reset(uint q, double rdoub);
void apply_reset(const bmReg& qs, std::vector<double> rdoubs);

void prob_filter(const bmReg& qs, const std::vector<uint>& q_mask, double &prob);

void setup(uint nq);
void clear();

};


