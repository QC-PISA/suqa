#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "complex_defines.cuh"


//const double TWOSQINV = 1./sqrt(2.);
#define TWOSQINV (1./sqrt(2.0))

namespace suqa{

extern uint blocks, threads;
extern cudaStream_t stream1, stream2;

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

void apply_x(ComplexVec& state, uint q);
void apply_x(ComplexVec& state, const std::vector<uint>& qs);

void apply_h(ComplexVec& state, uint q);
void apply_h(ComplexVec& state, const std::vector<uint>& qs);

//void apply_cx(ComplexVec& state, uint q_control, uint q_target);
void apply_cx(ComplexVec& state, const uint& q_control, const uint& q_target, const uint& q_mask=1U);

void apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const uint& q_target);
void apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const std::vector<uint>& q_mask, const uint& q_target);

void apply_swap(ComplexVec& state, const uint& q1, const uint& q2);


/* SUQA utils */
void measure_qbit(ComplexVec& state, uint q, uint& c, double rdoub);

void apply_reset(ComplexVec& state, uint q, double rdoub);
void apply_reset(ComplexVec& state, std::vector<uint> qs, std::vector<double> rdoubs);

};
