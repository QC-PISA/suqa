#pragma once
#include "suqa.cuh"
#include <omp.h>

void func_all_zeros(double* vec_data, size_t size) {
	for (uint i = 0; i < size; ++i)
		vec_data[i] = 0.0;

	vec_data[0] = 1.0;
}

double func_suqa_vnorm(double* vec_data, size_t size) {
// unthreaded
	double ret = 0.0;
	for (uint i = 0; i < size; ++i) {
		ret += vec_data[i] * vec_data[i];
	}
	return ret;
}

void func_suqa_vnormalize_by(double* v_comp, size_t len, double value) {
	for (uint i = 0; i < len; ++i) {
		v_comp[i] *= value;
	}
}

void func_suqa_x(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
    glob_mask |= (1U << q);
    for(uint i=0U; i<len;++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            double tmpval = state_re[i];
            state_re[i] = state_re[j];
            state_re[j] = tmpval;
            tmpval = state_im[i];
            state_im[i] = state_im[j];
            state_im[j] = tmpval;
        }
    }
}

void func_suqa_y(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
    glob_mask |= (1U << q);
    for(uint i=0U; i<len;++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
// j=0, i=1; ap[0] = -i*a[1]; ap[1]=i*a[0]
            double tmpval = state_re[i];
            state_re[i] = -state_im[j];
            state_im[j] = -tmpval;
            tmpval = state_im[i];
            state_im[i] = state_re[j];
            state_re[j] = tmpval;
        }
    }
}

void func_suqa_sigma_plus(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
    glob_mask |= (1U << q);
    for(uint i=0U;i<len; ++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            state_re[j] = state_re[i];
            state_im[j] = state_im[i];
            state_re[i] = 0;
            state_im[i] = 0;
        }
    }
}

void func_suqa_sigma_minus(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
    glob_mask |= (1U << q);
    for(uint i=0U;i<len; ++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            state_re[i] = state_re[j];
            state_im[i] = state_im[j];
            state_re[j] = 0;
            state_im[j] = 0;
        }
    }
}

void func_suqa_h(double* state_re, double* state_im, size_t len, uint q, uint glob_mask) {
    uint loc_mask = glob_mask | (1U << q);
    for(uint i_0=0U;i_0<len; ++i_0){
        if ((i_0 & loc_mask) == glob_mask) {
            const uint i_1 = i_0 | (1U << q);
            double a_0_re = state_re[i_0];
            double a_1_re = state_re[i_1];
            double a_0_im = state_im[i_0];
            double a_1_im = state_im[i_1];

            state_re[i_0] = TWOSQINV * (a_0_re + a_1_re);
            state_re[i_1] = TWOSQINV * (a_0_re - a_1_re);
            state_im[i_0] = TWOSQINV * (a_0_im + a_1_im);
            state_im[i_1] = TWOSQINV * (a_0_im - a_1_im);
        }
    }
}

void func_suqa_u1(double* state_re, double* state_im, size_t len, uint q, Complex phase, uint qmask, uint glob_mask) {
    //    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);

    glob_mask |= (1U << q);
    for(uint i=0U;i<len; ++i){
        if ((i & glob_mask) == qmask) {  // q_mask 
            double tmpval = state_re[i];
            state_re[i] = state_re[i] * phase.x - state_im[i] * phase.y;
            state_im[i] = tmpval * phase.y + state_im[i] * phase.x;

        }
    }
}

void func_suqa_mcx(double* const state_re, double* const state_im, size_t len, uint control_mask, uint mask_qs, uint q_target) {
    for(uint i=0U;i<len; ++i){
        if ((i & control_mask) == mask_qs) {
            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i];
            state_re[i] = state_re[j];
            state_re[j] = tmpval;
            tmpval = state_im[i];
            state_im[i] = state_im[j];
            state_im[j] = tmpval;
        }
    }
}
