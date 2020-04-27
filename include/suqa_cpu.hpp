#pragma once
#include "suqa.cuh"
#include <omp.h>

void func_suqa_init_state(double* vec_data, size_t size) {
	for (uint i = 0; i < size; ++i)
		vec_data[i] = 0.0;

	vec_data[0] = 1.0;
#ifdef SPARSE
    suqa::actives = std::vector<uint>(1, 0U);
#endif
}

double func_suqa_vnorm(double* vec_data, size_t size) {
// unthreaded
	double ret = 0.0;
#ifdef SPARSE
	for (const auto& idx : suqa::actives){
		ret += vec_data[idx+size/2] * vec_data[idx+size/2];
#else
    for (uint idx = 0U; idx < size; ++idx) {
#endif
		ret += vec_data[idx] * vec_data[idx];
	}
	return ret;
}

void func_suqa_vnormalize_by(double* v_comp, size_t len, double value) {
#ifdef SPARSE
	for (const auto& idx : suqa::actives){
		v_comp[idx+len/2] *= value;
#else
    for (uint idx = 0U; idx < size; ++idx) {
#endif
		v_comp[idx] *= value;
	}
}

inline void util_signed_swap(double* a, double* b, double sign1=1.0, double sign2=1.0) {
    // utility function for pauli rotation
    double cpy = *a;
    *a = (*b) * sign1;
    *b = cpy * sign2;
}

inline void util_rotate4(double* a, double* b, double* c, double* d, double ctheta, double stheta) {
    // utility function for pauli rotation
    double cpy = *a;
    *a = cpy * ctheta - (*b) * stheta;
    *b = (*b) * ctheta + cpy * stheta;
    cpy = *c;
    *c = cpy * ctheta - (*d) * stheta;
    *d = (*d) * ctheta + cpy * stheta;
}


void func_suqa_x(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
#ifdef SPARSE
    std::vector<uint> visited;
    for (auto& i: suqa::actives) { // number of actives is conserved
        if ((i& glob_mask)==glob_mask) { 
            uint j = i^ (1U << q); // flip of q bit
            if (std::find(visited.begin(),visited.end(),i)==visited.end()) { // swap only once
				util_signed_swap(&state_re[i], &state_re[j]);
				util_signed_swap(&state_im[i], &state_im[j]);
            } else {
                visited.push_back(i);
                visited.push_back(j);
            }
            i= j;
        }
    }
#else
    glob_mask |= (1U << q);
    for(uint i=0U; i<len;++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            util_signed_swap(&state_re[i], &state_re[j]);
            util_signed_swap(&state_im[i], &state_im[j]);
        }
    }
#endif
}

void func_suqa_y(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
#ifdef SPARSE
    std::vector<uint> visited;
    for (auto& i: suqa::actives) { // number of actives is conserved
        if ((i& glob_mask)==glob_mask) { 
            uint j = i^ (1U << q); // flip of q bit
            if (std::find(visited.begin(),visited.end(),i)==visited.end()) { // swap only once
				util_signed_swap(&state_re[i], &state_im[j], -1.0, -1.0);
				util_signed_swap(&state_im[i], &state_re[j]);
            } else {
                visited.push_back(i);
                visited.push_back(j);
            }
            i = j;
        }
    }
#else
    glob_mask |= (1U << q);
    for(uint i=0U; i<len;++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
			util_signed_swap(&state_re[i], &state_im[j], -1.0, -1.0);
			util_signed_swap(&state_im[i], &state_re[j]);
        }
    }
#endif
}

//void func_suqa_sigma_plus(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
//    glob_mask |= (1U << q);
//    for(uint i=0U;i<len; ++i){
//        if ((i & glob_mask) == glob_mask) {
//            uint j = i & ~(1U << q); // j has 0 on q-th digit
//            state_re[j] = state_re[i];
//            state_im[j] = state_im[i];
//            state_re[i] = 0;
//            state_im[i] = 0;
//        }
//    }
//}
//
//void func_suqa_sigma_minus(double* const state_re, double* const state_im, size_t len, uint q, uint glob_mask) {
//    glob_mask |= (1U << q);
//    for(uint i=0U;i<len; ++i){
//        if ((i & glob_mask) == glob_mask) {
//            uint j = i & ~(1U << q); // j has 0 on q-th digit
//            state_re[i] = state_re[j];
//            state_im[i] = state_im[j];
//            state_re[j] = 0;
//            state_im[j] = 0;
//        }
//    }
//}

void func_suqa_h(double* state_re, double* state_im, size_t len, uint q, uint glob_mask) {
#ifdef SPARSE
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for (const uint& i : suqa::actives) { // non-costant actives
        if ((i & glob_mask) == glob_mask) { // any matching the controls
            uint i_0 = i & ~(1U << q);
            uint i_1 = i_0 | (1U << q);
            if (std::find(visited.begin(), visited.end(), i_0) == visited.end()) { // apply only once
                double new_0_re = TWOSQINV * (state_re[i_0] + state_re[i_1]);
                double new_0_im = TWOSQINV * (state_im[i_0] + state_im[i_1]);
                double new_1_re = TWOSQINV * (state_re[i_0] - state_re[i_1]);
                double new_1_im = TWOSQINV * (state_im[i_0] - state_im[i_1]);

				state_re[i_0] = new_0_re;
				state_im[i_0] = new_0_im;
				state_re[i_1] = new_1_re;
				state_im[i_1] = new_1_im;
                if (norm(new_0_re, new_0_im) > 1e-8) {
                    // new_1 is ensured to have non zero contribution, since the pair was active 
                    new_actives.push_back(i_0);

                }
                if (norm(new_1_re, new_1_im) > 1e-8) {
                    new_actives.push_back(i_1);

                }
                visited.push_back(i_0);
            }
        } else { // add to new_actives except for i_1, which is already managed above
			new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
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
#endif
}

void func_suqa_u1(double* state_re, double* state_im, size_t len, uint q, Complex phase, uint qmask, uint glob_mask) {
    //    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);

    glob_mask |= (1U << q);
#ifdef SPARSE
    for (const auto& i : suqa::actives) { // actives is conserved
#else
    for(uint i=0U;i<len; ++i){
#endif
        if ((i & glob_mask) == qmask) {  // q_mask 
            double tmpval = state_re[i];
            state_re[i] = state_re[i] * phase.x - state_im[i] * phase.y;
            state_im[i] = tmpval * phase.y + state_im[i] * phase.x;
        }
    }
}

void func_suqa_mcx(double* const state_re, double* const state_im, size_t len, uint control_mask, uint mask_qs, uint q_target) {
#ifdef SPARSE
    std::vector<uint> visited;
    for (auto& i : suqa::actives) { // number of actives is conserved
        if ((i & control_mask) == mask_qs) {
            uint j = i ^ (1U << q_target); // flip of q bit
            if (std::find(visited.begin(),visited.end(),i)==visited.end()) { // swap only once
                std::swap(state_re[j], state_re[i]); // may not set to zero unactive (should always be overwritten)
                std::swap(state_im[j], state_im[i]);
                visited.push_back(i);
                visited.push_back(j);
            }
            i = j;
        }
    }
#else
    control_mask |= (1U << q_target);
    mask_qs |= (1U << q_target);
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
#endif
}

void func_suqa_mcu1(double* const state_re, double* const state_im, size_t len, uint control_mask, uint mask_qs, uint q_target, Complex rphase) {
#ifdef SPARSE
    for (auto& i : suqa::actives) { // number of actives is conserved
#else
    for(uint i=0U;i<len; ++i){
#endif
        if ((i & control_mask) == mask_qs) {
            //            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i];
            state_re[i] = state_re[i] * rphase.x - state_im[i] * rphase.y;
            state_im[i] = tmpval * rphase.y + state_im[i] * rphase.x;
        }
    }
}

void func_suqa_swap(double* const state_re, double* const state_im, size_t len, uint mask00, uint mask11, uint mask_q1, uint mask_q2) {
#ifdef SPARSE
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for (auto& i : suqa::actives) { // number of actives is conserved
        uint i_0 = i & ~mask_q1 & ~mask_q2;
        if (i & mask00){
            if(std::find(visited.begin(),visited.end(),i_0)==visited.end()) {
				// i -> ...00..., i_1 -> ...10..., i_2 -> ...01...
				uint i_1 = i_0 | mask_q1;
				uint i_2 = i_0 | mask_q2;
				uint i_3 = i_1 | i_2;
				util_signed_swap(&state_re[i_1], &state_re[i_2]);
				util_signed_swap(&state_im[i_1], &state_im[i_2]);
				if (norm(state_re[i_0], state_im[i_0]) > 1e-8) {
					new_actives.push_back(i_0);
				}
				if (norm(state_re[i_1], state_im[i_1]) > 1e-8) {
					new_actives.push_back(i_1);
				}
				if (norm(state_re[i_2], state_im[i_2]) > 1e-8) {
					new_actives.push_back(i_2);
				}
				if (norm(state_re[i_3], state_im[i_3]) > 1e-8) {
					new_actives.push_back(i_3);
				}
				visited.push_back(i_0);
        } else {
            new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
    for(uint i=0U;i<len; ++i){
        if ((i & mask11) == mask00) {
            // i -> ...00..., i_1 -> ...10..., i_2 -> ...01...
            uint i_1 = i | mask_q1;
            uint i_2 = i | mask_q2;
            double tmpval = state_re[i_1];
            state_re[i_1] = state_re[i_2];
            state_re[i_2] = tmpval;
            tmpval = state_im[i_1];
            state_im[i_1] = state_im[i_2];
            state_im[i_2] = tmpval;
        }
    }
#endif
}

//void func_suqa_phase_list(double* const state_re, double* const state_im, size_t len, std::vector<Complex> c_phases, uint mask0s, uint bm_offset, uint size_mask) {
//    for(uint i=0U;i<len; ++i){
//        if (i & mask0s) { // any state with gmask set
//            uint ph_idx = (i >> bm_offset) & size_mask; //index in the phases list
//            Complex cph = c_phases[ph_idx];
//            double tmpval = state_re[i];
//            state_re[i] = state_re[i] * cph.x - state_im[i] * cph.y;
//            state_im[i] = tmpval * cph.y + state_im[i] * cph.x;
//        }
//    }
//}
//
//// PAULI TENSOR PRODUCT ROTATIONS
//void func_suqa_pauli_TP_rotation_x(double* const state_re, double* const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta) {
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if ((i_0 & mask1s) == mask0s) {
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//
//            util_rotate4(&state_re[i_0], &state_im[i_1], &state_re[i_1], &state_im[i_0], ctheta, stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_y(double* const state_re, double* const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta) {
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if ((i_0 & mask1s) == mask0s) {
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//
//            util_rotate4(&state_re[i_0], &state_re[i_1], &state_im[i_0], &state_im[i_1], ctheta, -stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_z(double* const state_re, double* const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta) {
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if ((i_0 & mask1s) == mask0s) {
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//
//            util_rotate4(&state_re[i_0], &state_im[i_0], &state_im[i_1], &state_re[i_1], ctheta, stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_xx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            
//            // 0<->3
//            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,stheta);
//
//            // 1<->2
//            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_yy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//
//            // 0<->3 yy is real negative on 0,3
//            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,-stheta);
//
//            // 1<->2 yy is real positive on 1,2
//            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_zz(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            
//            // kl -> i(-)^(k+l) kl
//
//            // +i
//            util_rotate4(&state_re[i_0],&state_im[i_0],&state_re[i_3],&state_im[i_3],ctheta,stheta);
//
//            // -i
//            util_rotate4(&state_re[i_1],&state_im[i_1],&state_re[i_2],&state_im[i_2],ctheta,-stheta);
//
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_xy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            
//            // 0<->3
//            util_rotate4(&state_re[i_0],&state_re[i_3],&state_im[i_0],&state_im[i_3],ctheta,-stheta);
//
//            // 2<->1
//            util_rotate4(&state_re[i_2],&state_re[i_1],&state_im[i_2],&state_im[i_1],ctheta,-stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_zx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            
//            // ix on 0<->1
//            util_rotate4(&state_re[i_0],&state_im[i_1],&state_re[i_1],&state_im[i_0],ctheta,stheta);
//
//            // -ix on 2<->3
//            util_rotate4(&state_re[i_2],&state_im[i_3],&state_re[i_3],&state_im[i_2],ctheta,-stheta);
//
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_zy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            
//            // iy on 0<->1
//            util_rotate4(&state_re[i_0],&state_re[i_1],&state_im[i_0],&state_im[i_1],ctheta,-stheta);
//
//            // -iy on 2<->3
//            util_rotate4(&state_re[i_2],&state_re[i_3],&state_im[i_2],&state_im[i_3],ctheta,stheta);
//
//        }
//    }
//}
//
//
//void func_suqa_pauli_TP_rotation_zxx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            uint i_4 = i_0 | mask_q3;
//            uint i_5 = i_4 | i_1;
//            uint i_6 = i_4 | i_2;
//            uint i_7 = i_4 | i_3;
//
//            // 0<->3
//            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,stheta);
//
//            // 1<->2
//            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
//
//            // 4<->7
//            util_rotate4(&state_re[i_4],&state_im[i_7],&state_re[i_7],&state_im[i_4],ctheta,-stheta);
//
//            // 5<->6
//            util_rotate4(&state_re[i_5],&state_im[i_6],&state_re[i_6],&state_im[i_5],ctheta,-stheta);
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_zyy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            uint i_4 = i_0 | mask_q3;
//            uint i_5 = i_4 | i_1;
//            uint i_6 = i_4 | i_2;
//            uint i_7 = i_4 | i_3;
//
//
//            // 0<->3 zyy is real negative on 0,3
//            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,-stheta);
//
//            // 1<->2 zyy is real positive on 1,2
//            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
//
//            // 4<->7 zyy is real positive on 4,7
//            util_rotate4(&state_re[i_4],&state_im[i_7],&state_re[i_7],&state_im[i_4],ctheta,stheta);
//
//            // 5<->6 zyy is real negative on 5,6
//            util_rotate4(&state_re[i_5],&state_im[i_6],&state_re[i_6],&state_im[i_5],ctheta,-stheta);
//            
//        }
//    }
//}
//
//void func_suqa_pauli_TP_rotation_zzz(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
//    for (uint i_0 = 0U; i_0 < len; ++i_0) {
//        if((i_0 & mask1s) == mask0s){
//            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
//            uint i_1 = i_0 | mask_q1;
//            uint i_2 = i_0 | mask_q2;
//            uint i_3 = i_2 | i_1;
//            uint i_4 = i_0 | mask_q3;
//            uint i_5 = i_4 | i_1;
//            uint i_6 = i_4 | i_2;
//            uint i_7 = i_4 | i_3;
//
//            // +i on 0, 3, 5, 6
//            util_rotate4(&state_re[i_0],&state_im[i_0],&state_re[i_3],&state_im[i_3],ctheta,stheta);
//            util_rotate4(&state_re[i_5],&state_im[i_5],&state_re[i_6],&state_im[i_6],ctheta,stheta);
//
//            // -i on 1, 2, 4, 7
//            util_rotate4(&state_re[i_1],&state_im[i_1],&state_re[i_2],&state_im[i_2],ctheta,-stheta);
//            util_rotate4(&state_re[i_4],&state_im[i_4],&state_re[i_7],&state_im[i_7],ctheta,-stheta);
//        }
//    }
//}
//
//// sets amplitudes with value <val> in qubit <q> to zero
//// !! it leaves the state unnormalized !!
//void func_suqa_set_ampl_to_zero(double* state_re, double* state_im, size_t len, uint q, uint val) {
//    for (uint i = 0U; i < len; ++i) {
//        if (((i >> q) & 1U) == val) {
//            state_re[i] = 0.0;
//            state_im[i] = 0.0;
//        }
//    }
//}
//
////TODO: optimize with parallel reduce
//double func_suqa_prob1(double *v_re, double *v_im, size_t len, uint q){
//    double ret = 0.0;
//    double tmpval;
//    for (uint i = 0U; i < len; ++i) {
//        if(i & (1U << q)){
//            tmpval = v_re[i];
//            ret +=  tmpval*tmpval;
//            tmpval = v_im[i];
//            ret +=  tmpval*tmpval;
//        }
//    }
//    return ret;
//}
//
//double func_suqa_prob_filter(double *v_re, double *v_im, size_t len, uint mask_qs, uint mask){
//    double ret = 0.0;
//    double tmpval;
//    for (uint i = 0U; i < len; ++i) {
//        if((i & mask_qs) == mask){
//            tmpval = v_re[i];
//            ret +=  tmpval*tmpval;
//            tmpval = v_im[i];
//            ret +=  tmpval*tmpval;
//        }
//    }
//    return ret;
//}
