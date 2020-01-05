#pragma once
#include <iostream>
#include <stdio.h>
#include "complex_defines.cuh"


//const double TWOSQINV = 1./sqrt(2.);
#define TWOSQINV ((float)(1./sqrt(2.0)))

namespace suqa{

/* Utility procedures */
double vnorm(uint blocks, uint threads, const ComplexVec& v);
void vnormalize(uint blocks, uint threads, ComplexVec& v);

template<typename T>
void apply_2x2mat(T& x1, T& x2, const T& m11, const T& m12, const T& m21, const T& m22){

            T x1_next = m11 * x1 + m12 * x2;
            T x2_next = m21 * x1 + m22 * x2;
            x1 = x1_next;
            x2 = x2_next;
}


/* SUQA gates */
//
//void qi_reset(ComplexVec& state, const uint& q);
//void qi_reset(ComplexVec& state, const vector<uint>& qs);
//
void apply_x(uint blocks, uint threads, ComplexVec& state, uint q);
//void qi_x(ComplexVec& state, const vector<uint>& qs);
//
void apply_h(uint blocks, uint threads, ComplexVec& state, uint q);
//void qi_h(ComplexVec& state, const vector<uint>& qs);
//
void apply_cx(uint blocks, uint threads, ComplexVec& state, uint q_control, uint q_target);
//void qi_cx(ComplexVec& state, const uint& q_control, const uint& q_mask, const uint& q_target);
//
//void qi_mcx(ComplexVec& state, const vector<uint>& q_controls, const uint& q_target);
//void qi_mcx(ComplexVec& state, const vector<uint>& q_controls, const vector<uint>& q_mask, const uint& q_target);
//
//void qi_swap(ComplexVec& state, const uint& q1, const uint& q2);
//
};
