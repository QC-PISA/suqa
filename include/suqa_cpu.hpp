#pragma once
#include "suqa.cuh"
#include <omp.h>

void func_suqa_init_state(double* vec_data) {
	for (uint i = 1; i < 2*suqa::state.size(); ++i)
		vec_data[i] = 0.0;

	vec_data[0] = 1.0;
#ifdef SPARSE
    suqa::actives = std::vector<uint>(1, 0U);
#endif
}

void func_suqa_init_state_from_vec(double* vec_data, std::vector<double> re_coeff, std::vector<double> im_coeff) {
    uint ss=suqa::state.size();
	for (uint i = 0; i < 2*ss; ++i)
		vec_data[i] = (i<ss) ? re_coeff[i] : im_coeff[i-ss];

//	vec_data[0] = 1.0;
#ifdef SPARSE
    throw std::runtime_error("ERROR: sparse mode doesn't support initialization from vector state yet!");
    suqa::actives = std::vector<uint>(1, 0U);
#endif
}

double func_suqa_vnorm(double* vec_data) {
// unthreaded
    const uint size=suqa::state.size();
	double ret = 0.0;
#ifdef SPARSE
	for (const auto& idx : suqa::actives){
		ret += vec_data[idx+size] * vec_data[idx+size]; // imaginary part
#else
    for (uint idx = 0U; idx < 2*size; ++idx) {
#endif
		ret += vec_data[idx] * vec_data[idx];
	}
	return ret;
}

void func_suqa_vnormalize_by(double* vec_data, double value) {
    const uint size=suqa::state.size();
#ifdef SPARSE
	for (const auto& idx : suqa::actives){
		vec_data[idx+size] *= value;
#else
    for (uint idx = 0U; idx < 2*size; ++idx) {
#endif
		vec_data[idx] *= value;
	}
}

inline void util_signed_swap(double* a, double* b, double sign1=1.0, double sign2=1.0) {
    // utility function for pauli rotation
    double cpy = *a;
    *a = (*b) * sign1;
    *b = cpy * sign2;
}

inline void util_rotate2(double* a, double* b, double ctheta, double stheta) {
    // utility function for pauli rotation
    double cpy = *a;
    *a = cpy * ctheta - (*b) * stheta;
    *b = (*b) * ctheta + cpy * stheta;
}

inline void util_rotate4(double* a, double* b, double* c, double* d, double ctheta, double stheta) {
    // utility function for pauli rotation
    util_rotate2(a,b,ctheta,stheta);
    util_rotate2(c,d,ctheta,stheta);
//    double cpy = *a;
//    *a = cpy * ctheta - (*b) * stheta;
//    *b = (*b) * ctheta + cpy * stheta;
//    cpy = *c;
//    *c = cpy * ctheta - (*d) * stheta;
//    *d = (*d) * ctheta + cpy * stheta;
}


void func_suqa_x(double* const state_re, double* const state_im, uint q, uint glob_mask) {
#ifdef SPARSE
    std::vector<uint> visited;
    for (auto& i: suqa::actives) { // number of actives is conserved
        if ((i & glob_mask)==glob_mask) { 
            uint j = i^ (1U << q); // flip of q bit
            uint i_0=i & ~(1U << q);
            if (std::find(visited.begin(),visited.end(),i_0)==visited.end()) { // swap only once
				util_signed_swap(&state_re[i], &state_re[j]);
				util_signed_swap(&state_im[i], &state_im[j]);
                visited.push_back(i_0);
            }
            i = j;
        }
    }
#else
    glob_mask |= (1U << q);
    for(uint i=0U; i<suqa::state.size();++i){
        if ((i & glob_mask) == glob_mask) {
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            util_signed_swap(&state_re[i], &state_re[j]);
            util_signed_swap(&state_im[i], &state_im[j]);
        }
    }
#endif
}

void func_suqa_y(double* const state_re, double* const state_im, uint q, uint glob_mask) {
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
    for(uint i=0U; i<suqa::state.size();++i){
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

void func_suqa_h(double* state_re, double* state_im, uint q, uint glob_mask) {
#ifdef SPARSE
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for(const uint& i : suqa::actives){
        if ((i & glob_mask) == glob_mask) { // any matching the controls
            uint i_0 = i & ~(1U << q);
            uint i_1 = i_0 | (1U << q);
            if(std::find(visited.begin(), visited.end(), i_0) == visited.end()) { // apply only once
                double a_0_re = state_re[i_0];
                double a_0_im = state_im[i_0];
                double a_1_re = state_re[i_1];
                double a_1_im = state_im[i_1];

                state_re[i_0] = TWOSQINV * (a_0_re + a_1_re);
                state_im[i_0] = TWOSQINV * (a_0_im + a_1_im);
                state_re[i_1] = TWOSQINV * (a_0_re - a_1_re);
                state_im[i_1] = TWOSQINV * (a_0_im - a_1_im);

                if (norm(state_re[i_0], state_im[i_0]) > 1e-8) {
                    // new_1 is ensured to have non zero contribution, since the pair was active 
                    new_actives.push_back(i_0);

                }
                if (norm(state_re[i_1], state_im[i_1]) > 1e-8) {
                    new_actives.push_back(i_1);

                }
                visited.push_back(i_0);
            }
        }else{ // add to new_actives except for i_1, which is already managed above
			new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
    uint loc_mask = glob_mask | (1U << q);
    for(uint i_0=0U;i_0<suqa::state.size(); ++i_0){
        if ((i_0 & loc_mask) == glob_mask) {
            const uint i_1 = i_0 | (1U << q);
            double a_0_re = state_re[i_0];
            double a_0_im = state_im[i_0];
            double a_1_re = state_re[i_1];
            double a_1_im = state_im[i_1];

            state_re[i_0] = TWOSQINV * (a_0_re + a_1_re);
            state_im[i_0] = TWOSQINV * (a_0_im + a_1_im);
            state_re[i_1] = TWOSQINV * (a_0_re - a_1_re);
            state_im[i_1] = TWOSQINV * (a_0_im - a_1_im);

//            [b1r,b0r] = [[c, -s],[s,c]] . [a1r,a0r]
//            [b0r,b1r] = [[z, z],[-z,z]] . [a0r,a1r]
//            
        }
    }
#endif
}

void func_suqa_u1(double* state_re, double* state_im, uint q, Complex phase, uint qmask, uint glob_mask) {
    //    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);

    glob_mask |= (1U << q);
#ifdef SPARSE
    for (const auto& i : suqa::actives) { // actives is conserved
#else
    for(uint i=0U;i<suqa::state.size(); ++i){
#endif
        if ((i & glob_mask) == qmask) {  // q_mask 
            util_rotate2(&state_re[i],&state_im[i],phase.x,phase.y);
        }
    }
}

void func_suqa_mcx(double* const state_re, double* const state_im, uint control_mask, uint mask_qs, uint q_target) {
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
            i = j; // here changes the actives
        }
    }
#else
    control_mask |= (1U << q_target);
    mask_qs |= (1U << q_target);
    for(uint i=0U;i<suqa::state.size(); ++i){
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

void func_suqa_mcu1(double* const state_re, double* const state_im, uint control_mask, uint mask_qs, Complex rphase) {
#ifdef SPARSE
    for (auto& i : suqa::actives) { // number of actives is conserved
#else
    for(uint i=0U;i<suqa::state.size(); ++i){
#endif
        if ((i & control_mask) == mask_qs) {
            //            uint j = i & ~(1U << q_target);
            util_rotate2(&state_re[i],&state_im[i],rphase.x,rphase.y);
        }
    }
}

void func_suqa_swap(double* const state_re, double* const state_im, uint mask00, uint mask11, uint mask_q1, uint mask_q2) {
#ifdef SPARSE
    (void)mask11; // to ignore unused warning
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for (auto& i : suqa::actives) { // number of actives is conserved
        uint i_0 = i & ~(mask_q1|mask_q2);
        if ((i & mask00)==mask00){
            if (std::find(visited.begin(), visited.end(), i_0) == visited.end()) {
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
            }
        } else {
            new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
    for(uint i=0U;i<suqa::state.size(); ++i){
        if ((i & mask11) == mask00) {
            // i -> ...00..., i_1 -> ...10..., i_2 -> ...01...
            uint i_1 = i | mask_q1;
            uint i_2 = i | mask_q2;
            util_signed_swap(&state_re[i_1], &state_re[i_2]);
            util_signed_swap(&state_im[i_1], &state_im[i_2]);
//            double tmpval = state_re[i_1];
//            state_re[i_1] = state_re[i_2];
//            state_re[i_2] = tmpval;
//            tmpval = state_im[i_1];
//            state_im[i_1] = state_im[i_2];
//            state_im[i_2] = tmpval;
        }
    }
#endif
}

void func_suqa_phase_list(std::vector<Complex> c_phases, uint mask0s, uint bm_offset, uint size_mask) {
#if SPARSE
    for (const uint& i : suqa::actives) {
#else
    for(uint i=0U;i<suqa::state.size(); ++i){
#endif
        if (i & mask0s) { // any state with gmask set
            uint ph_idx = (i >> bm_offset) & size_mask; //index in the phases list
            Complex cph = c_phases[ph_idx];
            util_rotate2(&suqa::state.data_re[i],&suqa::state.data_im[i],cph.x,cph.y);
        }
    }
}

// PAULI TENSOR PRODUCT ROTATIONS

//TODO: get rid of mask1s if possible
void func_suqa_pauli_TP_rotation_pauli1(uint p1, double* const state_re, double* const state_im, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta) {
#ifdef SPARSE
    (void)mask1s;
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for (const uint& i : suqa::actives) {
        if ((i & mask0s) == mask0s) {
            // i_0 -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[4];
            i_[0] = i & ~mask_q1;
            i_[1] = i_[0] | mask_q1;
#else
    for (uint i_0 = 0U; i_0 < suqa::state.size(); ++i_0) {
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[4];
            i_[0] = i_0;
            i_[1] = i_[0] | mask_q1;
#endif
#ifdef SPARSE
            if (std::find(visited.begin(), visited.end(), i_[0]) == visited.end()) { // apply only once
#endif
            switch(p1){
            case 1: // X
                util_rotate4(&state_re[i_[0]], &state_im[i_[1]], &state_re[i_[1]], &state_im[i_[0]], ctheta, stheta);
                break;
            case 2: // Y
                util_rotate4(&state_re[i_[0]], &state_re[i_[1]], &state_im[i_[0]], &state_im[i_[1]], ctheta, -stheta);
                break;
            case 3: // Z
                util_rotate4(&state_re[i_[0]], &state_im[i_[0]], &state_im[i_[1]], &state_re[i_[1]], ctheta, stheta);
                break;
            }

#ifdef SPARSE
                for(size_t s=0; s<2; ++s){
                    if (norm(state_re[i_[s]], state_im[i_[s]]) > 1e-8)
                        new_actives.push_back(i_[s]);
                }

                visited.push_back(i_[0]);
            }
        } else { // add to new_actives except for i_[1] which is already managed above
			new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
        }
    }
#endif
}


void func_suqa_pauli_TP_rotation_pauli2(const uint p1, const uint p2, double *const state_re, double *const state_im, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
#ifdef SPARSE
    (void)mask1s;
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    std::sort(suqa::actives.begin(),suqa::actives.end());
    for (const uint& i : suqa::actives) {
        if ((i & mask0s) == mask0s) {
            // i_0 -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[4];
            i_[0] = i & ~(mask_q1|mask_q2);
#else
    for (uint i_0 = 0U; i_0 < suqa::state.size(); ++i_0) {
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[4];
            i_[0] = i_0;
#endif
            i_[1] = i_[0] | mask_q1;
            i_[2] = i_[0] | mask_q2;
            i_[3] = i_[2] | i_[1];

#ifdef SPARSE
            if (std::find(visited.begin(), visited.end(), i_[0]) == visited.end()) { // apply only once
#endif
//                std::cout<<"Before "<<i_[0]<<std::endl;
//                std::cout<<"state["<<i_[0]<<"] = "<<state_re[i_[0]]<<"+i*("<<state_im[i_[0]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[1]<<"] = "<<state_re[i_[1]]<<"+i*("<<state_im[i_[1]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[2]<<"] = "<<state_re[i_[2]]<<"+i*("<<state_im[i_[2]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[3]<<"] = "<<state_re[i_[3]]<<"+i*("<<state_im[i_[3]]<<")"<<std::endl;
                if(p1==1 and p2==1){ // XX
                    util_rotate4(&state_re[i_[0]],&state_im[i_[3]],&state_re[i_[3]],&state_im[i_[0]],ctheta,stheta); // 0<->3
                    util_rotate4(&state_re[i_[1]],&state_im[i_[2]],&state_re[i_[2]],&state_im[i_[1]],ctheta,stheta); // 1<->2
                }else if(p1==2 and p2==2){ // YY
                    util_rotate4(&state_re[i_[0]],&state_im[i_[3]],&state_re[i_[3]],&state_im[i_[0]],ctheta,-stheta); // 0<->3 yy is real negative on 0,3
                    util_rotate4(&state_re[i_[1]],&state_im[i_[2]],&state_re[i_[2]],&state_im[i_[1]],ctheta,stheta); // 1<->2 yy is real positive on 1,2
                }else if(p1==3 and p2==3){ // ZZ
                    util_rotate4(&state_re[i_[0]],&state_im[i_[0]],&state_re[i_[3]],&state_im[i_[3]],ctheta,stheta); // +i
                    util_rotate4(&state_re[i_[1]],&state_im[i_[1]],&state_re[i_[2]],&state_im[i_[2]],ctheta,-stheta); // -i
                }else if(p1==1 and p2==2){ // XY (least significant on the left)
                    util_rotate4(&state_re[i_[0]],&state_re[i_[3]],&state_im[i_[0]],&state_im[i_[3]],ctheta,-stheta); // 0<->3
                    util_rotate4(&state_re[i_[2]],&state_re[i_[1]],&state_im[i_[2]],&state_im[i_[1]],ctheta,-stheta); // 2<->1
                }else if(p1==1 and p2==3){ // XZ (least significant on the left)
                    util_rotate4(&state_re[i_[0]],&state_im[i_[1]],&state_re[i_[1]],&state_im[i_[0]],ctheta,stheta); // ix on 0<->1
                    util_rotate4(&state_re[i_[2]],&state_im[i_[3]],&state_re[i_[3]],&state_im[i_[2]],ctheta,-stheta); // -ix on 2<->3
                }else if(p1==2 and p2==3){ // YZ (least significant on the left)
                    util_rotate4(&state_re[i_[0]],&state_re[i_[1]],&state_im[i_[0]],&state_im[i_[1]],ctheta,-stheta); // iy on 0<->1
                    util_rotate4(&state_re[i_[2]],&state_re[i_[3]],&state_im[i_[2]],&state_im[i_[3]],ctheta,stheta); // -iy on 2<->3
                }
//                std::cout<<"After "<<i_[0]<<std::endl;
//                std::cout<<"state["<<i_[0]<<"] = "<<state_re[i_[0]]<<"+i*("<<state_im[i_[0]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[1]<<"] = "<<state_re[i_[1]]<<"+i*("<<state_im[i_[1]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[2]<<"] = "<<state_re[i_[2]]<<"+i*("<<state_im[i_[2]]<<")"<<std::endl;
//                std::cout<<"state["<<i_[3]<<"] = "<<state_re[i_[3]]<<"+i*("<<state_im[i_[3]]<<")"<<std::endl;
#ifdef SPARSE
                for(size_t s=0; s<4; ++s){
                    if (norm(state_re[i_[s]], state_im[i_[s]]) > 1e-8)
                        new_actives.push_back(i_[s]);
                }

                visited.push_back(i_[0]);
            }
        } else { // add to new_actives except for i_[1:3] which are already managed above
			new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
        }
    }
#endif
}

void func_suqa_pauli_TP_rotation_pauli3(const uint p1, const uint p2, const uint p3, double *const state_re, double *const state_im, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
#ifdef SPARSE
    (void)mask1s;
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
    for (const uint& i : suqa::actives) {
        if ((i & mask0s) == mask0s) {
            // i_0 -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[8];
            i_[0] = i & ~(mask_q1|mask_q2|mask_q3);
#else
    for (uint i_0 = 0U; i_0 < suqa::state.size(); ++i_0) {
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_[8];
            i_[0] = i_0;
#endif
            i_[1] = i_[0] | mask_q1;
            i_[2] = i_[0] | mask_q2;
            i_[3] = i_[2] | i_[1];
            i_[4] = i_[0] | mask_q3;
            i_[5] = i_[4] | i_[1];
            i_[6] = i_[4] | i_[2];
            i_[7] = i_[4] | i_[3];

#ifdef SPARSE
            if (std::find(visited.begin(), visited.end(), i_[0]) == visited.end()) { // apply only once
#endif
            if(p1==3 and p2==1 and p3==1){ // ZXX
                util_rotate4(&state_re[i_[0]],&state_im[i_[3]],&state_re[i_[3]],&state_im[i_[0]],ctheta,stheta); // 0<->3
                util_rotate4(&state_re[i_[1]],&state_im[i_[2]],&state_re[i_[2]],&state_im[i_[1]],ctheta,stheta); // 1<->2
                util_rotate4(&state_re[i_[4]],&state_im[i_[7]],&state_re[i_[7]],&state_im[i_[4]],ctheta,-stheta); // 4<->7
                util_rotate4(&state_re[i_[5]],&state_im[i_[6]],&state_re[i_[6]],&state_im[i_[5]],ctheta,-stheta); // 5<->6
            }else if(p1==3 and p2==2 and p3==2){ // ZYY
                util_rotate4(&state_re[i_[0]],&state_im[i_[3]],&state_re[i_[3]],&state_im[i_[0]],ctheta,-stheta); // 0<->3 zyy is real negative on 0,3
                util_rotate4(&state_re[i_[1]],&state_im[i_[2]],&state_re[i_[2]],&state_im[i_[1]],ctheta,stheta); // 1<->2 zyy is real positive on 1,2
                util_rotate4(&state_re[i_[4]],&state_im[i_[7]],&state_re[i_[7]],&state_im[i_[4]],ctheta,stheta); // 4<->7 zyy is real positive on 4,7
                util_rotate4(&state_re[i_[5]],&state_im[i_[6]],&state_re[i_[6]],&state_im[i_[5]],ctheta,-stheta); // 5<->6 zyy is real negative on 5,6
            }else if(p1==3 and p2==3 and p3==3){ // ZZZ
                util_rotate4(&state_re[i_[0]],&state_im[i_[0]],&state_re[i_[3]],&state_im[i_[3]],ctheta,stheta); // +i on 0, 3, 5, 6
                util_rotate4(&state_re[i_[5]],&state_im[i_[5]],&state_re[i_[6]],&state_im[i_[6]],ctheta,stheta);
                util_rotate4(&state_re[i_[1]],&state_im[i_[1]],&state_re[i_[2]],&state_im[i_[2]],ctheta,-stheta); // -i on 1, 2, 4, 7
                util_rotate4(&state_re[i_[4]],&state_im[i_[4]],&state_re[i_[7]],&state_im[i_[7]],ctheta,-stheta);
            }

#ifdef SPARSE
                for(size_t s=0; s<8; ++s){
                    if (norm(state_re[i_[s]], state_im[i_[s]]) > 1e-8)
                        new_actives.push_back(i_[s]);
                }

                visited.push_back(i_[0]);
            }
        } else { // add to new_actives except for i_[1:3] which are already managed above
			new_actives.push_back(i);
        }
    }
    suqa::actives.swap(new_actives);
#else
        }
    }
#endif
}


// sets amplitudes with value <val> in qubit <q> to zero
// !! it leaves the state unnormalized !!
void func_suqa_set_ampl_to_zero(double* state_re, double* state_im, uint q, uint val) {
#ifdef SPARSE
    auto it = suqa::actives.begin();
    while(it != suqa::actives.end()) {
        if(((*it >> q) & 1U) == val) {
            state_re[*it] = 0.0;
            state_im[*it] = 0.0;
            it = suqa::actives.erase(it);
        }else
            ++it;
    }
#else
    for (uint idx = 0U; idx < suqa::state.size(); ++idx) {
        if (((idx >> q) & 1U) == val) {
            state_re[idx] = 0.0;
            state_im[idx] = 0.0;
        }
    }
#endif
}

//TODO: optimize with parallel reduce
double func_suqa_prob1(double* const state_re, double* const state_im, uint q){
// unthreaded
    double ret = 0.0;
    double tmpval;

#ifdef SPARSE
	for (const auto& idx : suqa::actives){
#else
    for (uint idx = 0U; idx < suqa::state.size(); ++idx) {
#endif
//		ret += vec_data[idx+size/2] * vec_data[idx+size/2];
        if(idx & (1U << q)){
            tmpval = state_re[idx];
            ret +=  tmpval*tmpval;
            tmpval = state_im[idx];
            ret +=  tmpval*tmpval;
        }
	}
    return ret;
}

double func_suqa_prob_filter(double *state_re, double *state_im, uint mask_qs, uint mask){
    double ret = 0.0;
    double tmpval;
#ifdef SPARSE
	for (const auto& idx : suqa::actives){
#else
    for (uint idx = 0U; idx < suqa::state.size(); ++idx) {
#endif
        if((idx & mask_qs) == mask){
            tmpval = state_re[idx];
            ret +=  tmpval*tmpval;
            tmpval = state_im[idx];
            ret +=  tmpval*tmpval;
        }
    }
    return ret;
}
