#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "io.hpp"
#include "complex_defines.cuh"
#include <stdexcept>

#ifdef CUDA

#if !defined(NDEBUG) 
extern double *host_state_re, *host_state_im;
#define DEBUG_READ_STATE(state) {\
    cudaMemcpyAsync(host_state_re,state.data_re,state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream1); \
    cudaMemcpyAsync(host_state_im,state.data_im,state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream2); \
    cudaDeviceSynchronize(); \
    printf("vnorm = %.12lg\n",suqa::vnorm(state));\
    sparse_print((double*)host_state_re,(double*)host_state_im, state.size()); \
} 
#else
#define DEBUG_READ_STATE(state)
#endif

#else
#define DEBUG_READ_STATE(state)
#endif

//const double TWOSQINV = 1./sqrt(2.);
#define TWOSQINV 0.7071067811865475 

typedef std::vector<uint> bmReg;
#define PAULI_ID 0
#define PAULI_X  1
#define PAULI_Y  2
#define PAULI_Z  3


namespace suqa{


extern uint blocks, threads;
extern cudaStream_t stream1, stream2;

// global control mask:
// it applies every next operation 
// using it as condition (the user should make sure
// to use it only for operations not involving it)
extern uint gc_mask;

void activate_gc_mask(const bmReg& q_controls);
void deactivate_gc_mask();

/* Utility procedures */
double vnorm(const ComplexVec& v);
void vnormalize(ComplexVec& v);


//__host__ __device__ static __inline__
//void apply_2x2mat(Complex& x1, Complex& x2, const Complex& m11, const Complex& m12, const Complex& m21, const Complex& m22){
//            Complex x1_next = m11 * x1 + m12 * x2;
//            Complex x2_next = m21 * x1 + m22 * x2;
//            x1 = x1_next;
//            x2 = x2_next;
//}
//
//__host__ __device__ static __inline__
//void apply_2x2mat_doub(Complex& x1, Complex& x2, const double& m11, const double& m12, const double& m21, const double& m22){
//            Complex x1_next = m11 * x1 + m12 * x2;
//            Complex x2_next = m21 * x1 + m22 * x2;
//            x1 = x1_next;
//            x2 = x2_next;
//}

//__host__ __device__ static __inline__
//void swap_cmpx(Complex *const a, Complex *const b){
//    Complex tmp_c = *a;
//    *a = *b;
//    *b = tmp_c;
//}


/* SUQA gates */
//

// single qbit gates
void apply_x(ComplexVec& state, uint q);
void apply_x(ComplexVec& state, const bmReg& qs);

void apply_y(ComplexVec& state, uint q);
void apply_y(ComplexVec& state, const bmReg& qs);

void apply_z(ComplexVec& state, uint q);
void apply_z(ComplexVec& state, const bmReg& qs);

void apply_sigmap(ComplexVec& state, uint q);
void apply_sigmap(ComplexVec& state, const bmReg& qs);

void apply_sigmam(ComplexVec& state, uint q);
void apply_sigmam(ComplexVec& state, const bmReg& qs);

void apply_h(ComplexVec& state, uint q);
void apply_h(ComplexVec& state, const bmReg& qs);

void apply_t(ComplexVec& state, uint q);
void apply_t(ComplexVec& state, const bmReg& qs);

void apply_tdg(ComplexVec& state, uint q);
void apply_tdg(ComplexVec& state, const bmReg& qs);

void apply_u1(ComplexVec& state, uint q, double phase);

// multiple qbit gates
//void apply_cx(ComplexVec& state, uint q_control, uint q_target);
void apply_cx(ComplexVec& state, const uint& q_control, const uint& q_target, const uint& q_mask=1U);

void apply_mcx(ComplexVec& state, const bmReg& q_controls, const uint& q_target);
void apply_mcx(ComplexVec& state, const bmReg& q_controls, const bmReg& q_mask, const uint& q_target);

void apply_cu1(ComplexVec& state, uint q_control, uint q_target, double phase, uint q_mask=1U);

void apply_mcu1(ComplexVec& state, const bmReg& q_controls, const uint& q_target, double phase);
void apply_mcu1(ComplexVec& state, const bmReg& q_controls, const bmReg& q_mask, const uint& q_target, double phase);

void apply_swap(ComplexVec& state, const uint& q1, const uint& q2);

// rotation by phase in the direction of a pauli tensor product
void apply_pauli_TP_rotation(ComplexVec& state, const bmReg& q_apply, const std::vector<uint>& pauli_TPconst, double phase);

/* SUQA utils */
void measure_qbit(ComplexVec& state, uint q, uint& c, double rdoub);
void measure_qbits(ComplexVec& state, const bmReg& qs, std::vector<uint>& cs,const std::vector<double>& rdoubs);


void apply_reset(ComplexVec& state, uint q, double rdoub);
void apply_reset(ComplexVec& state, const bmReg& qs, std::vector<double> rdoubs);

void setup(uint Dim);
void clear();

void prob_filter(ComplexVec& state, const bmReg& qs, const std::vector<uint>& q_mask, double &prob);

};

