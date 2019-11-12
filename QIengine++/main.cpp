#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <bits/stdc++.h>
#include <cmath>
#include <cassert>
#include "include/Rand.hpp"
#include <chrono>
#include "include/parser.hpp"
#include "include/suqa_gates.hpp"

#ifndef NDEBUG
    #define DEBUG_CALL(x) x
#else
    #define DEBUG_CALL(x)
#endif

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
"\nSimulator for Universal Quantum Algorithms\n" 
"          by Giuseppe Clemente            \n"); 
}



/* Hamiltonian
 *
 * H = 1/4 (1 + X1 X0 + X2 X0 + X2 X1)
 *
 */


// simulation parameters
double beta;
double f1;
double f2;
const uint nqubits = 6;
const uint Dim = (uint)pow(2.0, nqubits); // simulation hyperparameters
uint max_reverse_attempts;
uint metro_steps;
uint reset_each;
unsigned long long iseed = 0ULL;
double t_phase_estimation;
int n_phase_estimation;
string Xmatstem="";

uint gCi;
uint c_acc = 0;


// Global state of the system.
// Ordering (less to most significant)
// psi[0], psi[1], E_old[0], E_old[1], E_new[0], E_new[1], acc //, qaux[0]
vector<Complex> gState(Dim,0.0);

//vector<double> energy_measures;
vector<double> X_measures;
vector<double> E_measures;

// Operator X parameter
const double phi = (1.+sqrt(5.))/2.;
const double mphi_inv = -1./phi;
const double Sa = phi/sqrt(2.+phi);
const double Sb = 1/sqrt(2.+phi);
const double S_10=Sa, S_12=Sb, S_20=-Sb, S_22=Sa;


// constants
const Complex rphase_m[3] = {exp((2*M_PI)*iu), exp((2*M_PI/2.)*iu), exp((2*M_PI/4.)*iu)};

// Utilities

pcg rangen;

// bit masks
enum bm_idxs {  bm_psi0, 
                bm_psi1,
                bm_psi2,
                bm_E_old0,
                bm_E_new0,
                bm_acc};


std::ostream& operator<<(std::ostream& s, const Complex& c){
    s<<"("<<real(c)<<", "<<imag(c)<<")";
    return s;
}

// Simulation procedures

void check_unused(){
    ;
//    uint mask1 = 3U;
//    uint mask2 = 12U;
//    for(uint i = 0U; i < gState.size(); ++i){
//        if( (i & mask1) == mask1){
//            assert(norm(gState[i])<1e-8);
//        }
//        if( (i & mask2) == mask2){
//            assert(norm(gState[i])<1e-8);
//        }
//    } 
}

void reset_non_state_qbits(){
    DEBUG_CALL(cout<<"\n\nBefore reset"<<endl);
    DEBUG_CALL(sparse_print(gState));
    qi_reset(gState, {bm_E_old0, bm_E_new0, bm_acc});
    DEBUG_CALL(cout<<"\n\nAfter reset"<<endl);
    DEBUG_CALL(sparse_print(gState));
}


void measure_qbit(vector<Complex>& state, const uint& q, uint& c){
    double prob1 = 0.0;

    for(uint i = 0U; i < state.size(); ++i){
        if((i >> q) & 1U){
            prob1+=norm(state[i]); 
        }
    }
    c = (uint)(rangen.doub() < prob1); // prob1=1 -> c = 1 surely
    
    if(c){ // set to 0 coeffs with bm_acc 0
        for(uint i = 0U; i < state.size(); ++i){
            if(((i >> q) & 1U) == 0U)
                state[i] = {0.0, 0.0};        
        }
    }else{ // set to 0 coeffs with bm_acc 1
        for(uint i = 0U; i < state.size(); ++i){
            if(((i >> q) & 1U) == 1U)
                state[i] = {0.0, 0.0};        
        }
    }
    vnormalize(state);
}

//TODO: can be optimized for multiple qbits measures?
void measure_qbits(vector<Complex>& state, const vector<uint>& qs, vector<uint>& cs){
    for(uint k = 0U; k < qs.size(); ++k)
        measure_qbit(state, qs[k], cs[k]);
}

void qi_crm(vector<Complex>& state, const uint& q_control, const uint& q_target, const int& m){
    for(uint i = 0U; i < state.size(); ++i){
        // for the swap, not only q_target:1 but also q_control:1
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            state[i] *= (m>0) ? rphase_m[m] : conj(rphase_m[-m]);
        }
    }
}

void qi_cu_on2(vector<Complex>& state, const double& dt, const uint& q_control, const vector<uint>& qstate){
    uint cmask = (1U << q_control);
	uint mask = cmask; // (1U << qstate[0]) | (1U << qstate[0])
    for(const auto& qs : qstate){
        mask |= (1U << qs);
    }

	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & mask) == cmask){
      
            uint i_1 = i_0 | (1U << qstate[0]);
            uint i_2 = i_0 | (1U << qstate[1]);
            uint i_3 = i_1 | i_2;

            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            Complex a_2 = state[i_2];
            Complex a_3 = state[i_3];
            
            state[i_0] = exp(-dt*iu)*a_0;
            state[i_1] = exp(-dt*iu)*(cos(dt)*a_1 -sin(dt)*iu*a_2);
            state[i_2] = exp(-dt*iu)*(-sin(dt)*iu*a_1 + cos(dt)*a_2);
            state[i_3] = exp(-dt*iu)*a_3;
        }
    }

}

void qi_cu_on3(vector<Complex>& state, const double& dt, const uint& q_control, const vector<uint>& qstate){
    uint cmask = (1U << q_control);
	uint mask = cmask; // (1U << qstate[0]) | (1U << qstate[0])
    for(const auto& qs : qstate){
        mask |= (1U << qs);
    }

	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & mask) == cmask){
      
            uint i_1 = i_0 | (1U << qstate[0]);
            uint i_2 = i_0 | (1U << qstate[1]);
            uint i_3 = i_1 | i_2;
            uint i_4 = i_0 | (1U << qstate[2]);
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;


            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            Complex a_2 = state[i_2];
            Complex a_3 = state[i_3];
            Complex a_4 = state[i_4];
            Complex a_5 = state[i_5];
            Complex a_6 = state[i_6];
            Complex a_7 = state[i_7];

            double dtp = dt/4.; 
            // apply 1/.4 (Id +X2 X1)
            state[i_0] = exp(-dtp*iu)*(cos(dtp)*a_0 -sin(dtp)*iu*a_6);
            state[i_1] = exp(-dtp*iu)*(cos(dtp)*a_1 -sin(dtp)*iu*a_7);
            state[i_2] = exp(-dtp*iu)*(cos(dtp)*a_2 -sin(dtp)*iu*a_4);
            state[i_3] = exp(-dtp*iu)*(cos(dtp)*a_3 -sin(dtp)*iu*a_5);
            state[i_4] = exp(-dtp*iu)*(cos(dtp)*a_4 -sin(dtp)*iu*a_2);
            state[i_5] = exp(-dtp*iu)*(cos(dtp)*a_5 -sin(dtp)*iu*a_3);
            state[i_6] = exp(-dtp*iu)*(cos(dtp)*a_6 -sin(dtp)*iu*a_0);
            state[i_7] = exp(-dtp*iu)*(cos(dtp)*a_7 -sin(dtp)*iu*a_1);

            a_0 = state[i_0];
            a_1 = state[i_1];
            a_2 = state[i_2];
            a_3 = state[i_3];
            a_4 = state[i_4];
            a_5 = state[i_5];
            a_6 = state[i_6];
            a_7 = state[i_7];

            // apply 1/.4 (X2 X0)
            state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_5);
            state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_4);
            state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_7);
            state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_6);
            state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_1);
            state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_0);
            state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_3);
            state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_2);

            a_0 = state[i_0];
            a_1 = state[i_1];
            a_2 = state[i_2];
            a_3 = state[i_3];
            a_4 = state[i_4];
            a_5 = state[i_5];
            a_6 = state[i_6];
            a_7 = state[i_7];

            // apply 1/.4 (X1 X0)
            state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_3);
            state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_2);
            state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_1);
            state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_0);
            state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_7);
            state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_6);
            state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_5);
            state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_4);
        }
    }

}

// TODO: generalize to higher D
void qi_qft(vector<Complex>& state, const vector<uint>& qact){
    if(qact.size()==1)
        qi_h(state, qact[0]);
    else if(qact.size()==2){
        qi_h(state, qact[1]);
        qi_crm(state, qact[0], qact[1], -2);
        qi_h(state, qact[0]);
    }else{
        throw std::runtime_error("ERROR: qft(inverse) not implemented for nqubits !=1,2");
    }
}

void qi_qft_inverse(vector<Complex>& state, const vector<uint>& qact){
    qi_qft(state, qact);
}

void apply_phase_estimation(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(cout<<"apply_phase_estimation()"<<endl);
    qi_h(state,q_target);
    DEBUG_CALL(cout<<"after qi_h(state,q_target)"<<endl);
    DEBUG_CALL(sparse_print(state));

    // apply CUs
    double dt = t/(double)n;

    for(int trg = q_target.size() - 1; trg > -1; --trg){
        for(uint ti = 0; ti < n; ++ti){
            for(uint itrs = 0; itrs < q_target.size()-trg; ++itrs){
                qi_cu_on3(state, dt, q_target[trg], q_state);
            }
        }
    }
    DEBUG_CALL(cout<<"\nafter evolutions"<<endl);
    DEBUG_CALL(sparse_print(state));
    
    // apply QFT^{-1}
    qi_qft_inverse(state, q_target); 

}

void apply_phase_estimation_inverse(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(cout<<"apply_phase_estimation_inverse()"<<endl);

    // apply QFT
    qi_qft(state, q_target); 


    // apply CUs
    double dt = t/(double)n;

    for(uint trg = 0; trg < q_target.size(); ++trg){
        for(uint ti = 0; ti < n; ++ti){
            for(uint itrs = 0; itrs < q_target.size()-trg; ++itrs){
                qi_cu_on3(state, -dt, q_target[trg], q_state);
            }
        }
    }
    
    qi_h(state,q_target);

}


void apply_Phi_old(){

    apply_phase_estimation(gState, {bm_psi0, bm_psi1, bm_psi2}, {bm_E_old0}, t_phase_estimation, n_phase_estimation);

}

void apply_Phi_old_inverse(){

    apply_phase_estimation_inverse(gState, {bm_psi0, bm_psi1, bm_psi2}, {bm_E_old0}, t_phase_estimation, n_phase_estimation);

}

void apply_Phi(){

    apply_phase_estimation(gState, {bm_psi0, bm_psi1, bm_psi2}, {bm_E_new0}, t_phase_estimation, n_phase_estimation);

}

void apply_Phi_inverse(){

    apply_phase_estimation_inverse(gState, {bm_psi0, bm_psi1, bm_psi2}, {bm_E_new0}, t_phase_estimation, n_phase_estimation);

}


uint draw_C(){
    double extract = rangen.doub();
    if (extract<1./3)
        return 0U;
    else if(extract<2./3)
        return 1U;
    else
        return 2U;
}

void apply_C(const uint &Ci){
    if(Ci==0U){
        qi_h(gState,bm_psi0);
    }else if(Ci==1U){
        qi_h(gState,bm_psi1);
    }else if(Ci==2U){
        qi_h(gState,bm_psi2);
    }else{
        throw std::runtime_error("Error!");
    }
}

void apply_C_inverse(const uint &Ci){
    apply_C(Ci);
}

const uint anc_regs = (1U <<bm_E_old0)  |(1U <<bm_E_new0) |(1U <<bm_acc);
//const uint E_old_acc_regs = (1U <<bm_E_old0) |(1U <<bm_acc);
const uint E_new_acc_regs = (1U <<bm_E_new0) |(1U <<bm_acc);

void apply_W(){
    DEBUG_CALL(cout<<"\n\nApply W"<<endl);
    // 
    uint mask = anc_regs;
    // Ei = 0, Ek = 1
    //(1U <<bm_E_new0) |(1U <<bm_acc);
    uint case1a = E_new_acc_regs;
    for(uint i = 0U; i < gState.size(); ++i){
        if(((i & mask) == case1a)){
            uint j = i & ~(1U << bm_acc);
            
            DEBUG_CALL(if(norm(gState[i])+norm(gState[j])>1e-8) cout<<"case1: gState["<<i<<"] = "<<gState[i]<<", gState["<<j<<"] = "<<gState[j]<<endl);
            apply_2x2mat(gState[j], gState[i], sqrt(1.-f1), sqrt(f1), sqrt(f1), -sqrt(1.-f1));
            DEBUG_CALL(if(norm(gState[i])+norm(gState[j])>1e-8) cout<<"after: gState["<<i<<"] = "<<gState[i]<<", gState["<<j<<"] = "<<gState[j]<<endl);
        }else if((i >> bm_acc) & 1U){
            uint j = i & ~(1U << bm_acc);

            DEBUG_CALL(if(norm(gState[i])+norm(gState[j])>1e-8) cout<<"case3: gState["<<i<<"] = "<<gState[i]<<", gState["<<j<<"] = "<<gState[j]<<endl);
            std::swap(gState[i],gState[j]);
            DEBUG_CALL(if(norm(gState[i])+norm(gState[j])>1e-8) cout<<"after: gState["<<i<<"] = "<<gState[i]<<", gState["<<j<<"] = "<<gState[j]<<endl);
        }
    }
}

void apply_W_inverse(){
    apply_W();
}

void apply_U(){
    DEBUG_CALL(cout<<"\n\nApply U"<<endl);
    apply_C(gCi);
    DEBUG_CALL(cout<<"\n\nAfter apply C = "<<gCi<<endl);
    DEBUG_CALL(sparse_print(gState));
    apply_Phi();
    DEBUG_CALL(cout<<"\n\nAfter second phase estimation"<<endl);
    DEBUG_CALL(sparse_print(gState));
    apply_W();
    DEBUG_CALL(cout<<"\n\nAfter apply W"<<endl);
    DEBUG_CALL(sparse_print(gState));
}

void apply_U_inverse(){
    apply_W_inverse();
    DEBUG_CALL(cout<<"\n\nAfter apply W inverse"<<endl);
    DEBUG_CALL(sparse_print(gState));
    apply_Phi_inverse();
    DEBUG_CALL(cout<<"\n\nAfter inverse second phase estimation"<<endl);
    DEBUG_CALL(sparse_print(gState));
    apply_C_inverse(gCi);
    DEBUG_CALL(cout<<"\n\nAfter apply C inverse = "<<gCi<<endl);
    DEBUG_CALL(sparse_print(gState));
}

Complex SXmat[8][8];

double measure_X(){
    if(Xmatstem==""){
        return 0.0;
    }

	uint mask = 7U;
	vector<uint> classics(3);

    vector<double> vals(8);

    FILE * fil_re = fopen((Xmatstem+"_vecs_re").c_str(),"r"); 
    FILE * fil_im = fopen((Xmatstem+"_vecs_im").c_str(),"r"); 
    FILE * fil_vals = fopen((Xmatstem+"_vals").c_str(),"r"); 
    double tmp_re,tmp_im;
    for(int i=0; i<8; ++i){
        fscanf(fil_vals, "%lg",&vals[i]);
//        cout<<"vals = "<<vals[i]<<endl;
        for(int j=0; j<8; ++j){
            fscanf(fil_re, "%lg",&tmp_re);
            fscanf(fil_im, "%lg",&tmp_im);
            SXmat[i][j] = tmp_re+tmp_im*iu;
//            cout<<real(SXmat[i][j])<<imag(SXmat[i][j])<<" ";
        }
        fscanf(fil_re, "\n");
        fscanf(fil_im, "\n");
//        cout<<endl;
    }

    fclose(fil_vals);
    fclose(fil_re);
    fclose(fil_im);

	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
      
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;
            uint i_3 = i_0 | 3U;
            uint i_4 = i_0 | 4U;
            uint i_5 = i_0 | 5U;
            uint i_6 = i_0 | 6U;
            uint i_7 = i_0 | 7U;


            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            Complex a_3 = gState[i_3];
            Complex a_4 = gState[i_4];
            Complex a_5 = gState[i_5];
            Complex a_6 = gState[i_6];
            Complex a_7 = gState[i_7];

            vector<uint> iss = {i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7};
            vector<Complex> ass = {a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7};


            for(int r=0; r<8; ++r){
                gState[iss[r]]=0.0;
                for(int c=0; c<8; ++c){
                     gState[iss[r]] += SXmat[r][c]*ass[c];
                }
            }
            

//            gState[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_6);
//            gState[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_7);
//            gState[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_4);
//            gState[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_5);
//            gState[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_2);
//            gState[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_3);
//            gState[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_0);
//            gState[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_1);

        }
    }
    measure_qbits(gState, {bm_psi0,bm_psi1,bm_psi2}, classics);

	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
      
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;
            uint i_3 = i_0 | 3U;
            uint i_4 = i_0 | 4U;
            uint i_5 = i_0 | 5U;
            uint i_6 = i_0 | 6U;
            uint i_7 = i_0 | 7U;


            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            Complex a_3 = gState[i_3];
            Complex a_4 = gState[i_4];
            Complex a_5 = gState[i_5];
            Complex a_6 = gState[i_6];
            Complex a_7 = gState[i_7];

            vector<uint> iss = {i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7};
            vector<Complex> ass = {a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7};

            for(int r=0; r<8; ++r){
                gState[iss[r]]=0.0;
                for(int c=0; c<8; ++c){
                     gState[iss[r]] += conj(SXmat[c][r])*ass[c];
                }
            }
        }
    }

    uint meas = classics[0] + 2*classics[1] + 4*classics[2];
    return vals[meas];
//    switch(meas){
//        case 0:
//            return vals[0];
//            break;
//        case 1:
//            return phi;
//            break;
//        case 2:
//            return mphi_inv;
//            break;
//        default:
//            throw "Error!";
//    }
//    return 0.0;
}

// double measure_X(){
// 	vector<uint> classics(2);
//     measure_qbits(gState, {bm_psi0,bm_psi1}, classics);
//     uint meas = classics[0] + 2*classics[1];
//     switch(meas){
//         case 0:
//             return 1.0;
//             break;
//         case 1:
//             return 2.0;
//             break;
//         case 2:
//             return 3.0;
//             break;
//         default:
//             throw "Error!";
//     }
//     return 0.0;
// }


void metro_step(uint s){
    DEBUG_CALL(cout<<"initial state"<<endl);
    DEBUG_CALL(sparse_print(gState));
    reset_non_state_qbits();
    DEBUG_CALL(check_unused());
    DEBUG_CALL(cout<<"state after reset"<<endl);
    DEBUG_CALL(sparse_print(gState));
    apply_Phi_old();
    DEBUG_CALL(check_unused());
    DEBUG_CALL(cout<<"\n\nAfter first phase estimation"<<endl);
    DEBUG_CALL(sparse_print(gState));

    gCi = draw_C();
    DEBUG_CALL(cout<<"\n\ndrawn C = "<<gCi<<endl);
    apply_U();
    DEBUG_CALL(check_unused());

    measure_qbit(gState, bm_acc, c_acc);

    if (c_acc == 1U){
        DEBUG_CALL(cout<<"accepted"<<endl);
        vector<uint> c_E_news(1,0), c_E_olds(1,0);
        measure_qbits(gState, {bm_E_new0}, c_E_news);
        DEBUG_CALL(double tmp_E=c_E_news[0]);
        DEBUG_CALL(cout<<"  energy measure : "<<tmp_E<<endl); 
        apply_Phi_inverse();
        if(s>0U and s%reset_each ==0U){
            E_measures.push_back(c_E_news[0]);
            qi_reset(gState, {bm_E_new0});
            X_measures.push_back(measure_X());
////            X_measures.push_back(0.0);
            DEBUG_CALL(cout<<"  X measure : "<<X_measures.back()<<endl); 
            DEBUG_CALL(cout<<"\n\nAfter X measure"<<endl);
            DEBUG_CALL(sparse_print(gState));
            DEBUG_CALL(cout<<"  X measure : "<<X_measures.back()<<endl); 
//            reset_non_state_qbits();
            qi_reset(gState, {bm_E_new0});
            apply_Phi();
            measure_qbits(gState, {bm_E_new0}, c_E_news);
            DEBUG_CALL(cout<<"\n\nAfter E recollapse"<<endl);
            DEBUG_CALL(sparse_print(gState));
            apply_Phi_inverse();
      }

        return;
    }
    //else

    DEBUG_CALL(cout<<"rejected; restoration cycle:"<<endl);
    apply_U_inverse();

    DEBUG_CALL(cout<<"\n\nBefore reverse attempts"<<endl);
    DEBUG_CALL(sparse_print(gState));
    uint iters = max_reverse_attempts;
    while(iters > 0){
        apply_Phi();
        double Eold_meas, Enew_meas;
        vector<uint> c_E_olds(1,0), c_E_news(1,0);
        measure_qbits(gState, {bm_E_old0}, c_E_olds);
        Eold_meas = c_E_olds[0];
        measure_qbits(gState, {bm_E_new0}, c_E_news);
        Enew_meas = c_E_news[0];
        apply_Phi_inverse();
        
        if(Eold_meas == Enew_meas){
//            E_measures.push_back(Eold_meas);
            DEBUG_CALL(cout<<"  accepted restoration ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<endl); 
            if(s>0U and s%reset_each == 0U){
                E_measures.push_back(Eold_meas);
                DEBUG_CALL(cout<<"  energy measure : "<<Eold_meas<<endl); 
                DEBUG_CALL(cout<<"\n\nBefore X measure"<<endl);
                DEBUG_CALL(sparse_print(gState));
                qi_reset(gState, {bm_E_new0});
                X_measures.push_back(measure_X());
////                X_measures.push_back(0.);
                DEBUG_CALL(cout<<"\n\nAfter X measure"<<endl);
                DEBUG_CALL(sparse_print(gState));
                DEBUG_CALL(cout<<"  X measure : "<<X_measures.back()<<endl); 
 ////               reset_non_state_qbits();
                qi_reset(gState, {bm_E_new0});
                apply_Phi();
                measure_qbits(gState, {bm_E_new0}, c_E_news);
                DEBUG_CALL(cout<<"\n\nAfter E recollapse"<<endl);
                DEBUG_CALL(sparse_print(gState));
                apply_Phi_inverse();
            }
            break;
        }
        //else
        DEBUG_CALL(cout<<"  rejected ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<endl); 
        uint c_acc_trash;
        apply_U(); 
        measure_qbit(gState, bm_acc, c_acc_trash); 
        apply_U_inverse(); 

        iters--;
    }

    if (iters == 0){
        cout<<"not converged :("<<endl;
        exit(1);
    }
}



int main(int argc, char** argv){
    if(argc < 5){
        cout<<"arguments: <beta> <metro steps> <reset each> <output file path> [--max-reverse <max reverse attempts>=20] [--seed <seed>=random] [--PE-time <time of PE evolution>=pi/2] [--PE-steps <steps of PE evolution>=10] [--X-mat-stem <stem for X measure matrix>]"<<endl;
        exit(1);
    }

    parse_arguments(args, argc, argv);

    beta = args.beta;
    metro_steps = (uint)args.metro_steps;
    reset_each = (uint)args.reset_each;
    string outfilename(args.outfile);
    max_reverse_attempts = (uint)args.max_reverse_attempts;
    t_phase_estimation = args.pe_time;
    n_phase_estimation = args.pe_steps;
    Xmatstem = args.Xmatstem;
    iseed = args.seed;
    if(iseed>0)
        rangen.set_seed(iseed);
    
    iseed = rangen.get_seed();

    f1 = exp(-beta);
    f2 = exp(-2.*beta);

    
    // Banner
    print_banner();
    cout<<"arguments:\n"<<args<<endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Initialization:
    // known eigenstate of the system: psi=0, E_old = 0
    
    std::fill_n(gState.begin(), gState.size(), 0.0);
    gState[0] = TWOSQINV; 
    gState[3] = -TWOSQINV; 
    for(uint s = 0U; s < metro_steps; ++s){
        metro_step(s);
    }

    cout<<"all fine :)\n"<<endl;

    FILE * fil = fopen(outfilename.c_str(), "w");

    fprintf(fil, "# it E X\n");

    for(uint ei = 0; ei < E_measures.size(); ++ei){
        fprintf(fil, "%d %.16lg %.16lg\n", ei, E_measures[ei], X_measures[ei]);
    }
//    for(uint ei = 0; ei < E_measures.size(); ++ei){
//        fprintf(fil, "%d %.16lg\n", ei, E_measures[ei]);
//    }
    fclose(fil);

    cout<<"\n\tSuqa!\n"<<endl;

    // test gates:
//    {
//        cout<<"TEST GATES"<<endl;
//        vector<Complex> test_state = {{0.4,-1.6},{1.2,0.7},{-0.1,0.6},{-1.3,0.4},{1.2,-1.3},{-1.2,1.7},{-3.1,0.6},{-0.3,0.2}};
//        vnormalize(test_state);
//        cout<<"initial state:"<<endl;
//        print(test_state);
//        cout<<"apply X to qbit 1 (most significant one)"<<endl;
//        qi_x(test_state, 1);
//        print(test_state);
//        cout<<"reapply X to qbit 1 (most significant one)"<<endl;
//        qi_x(test_state, 1);
//        print(test_state);
//        cout<<"apply CX controlled by qbit 0 to qbit 1"<<endl;
//        qi_cx(test_state, 0, 1);
//        print(test_state);
//        cout<<"apply CCX controlled by qbit 1 and 2 to qbit 0"<<endl;
//        qi_mcx(test_state, {1,2}, 0);
//        print(test_state);
//    }
//    { 
//        cout<<"\nTEST SIMULATION"<<endl;
//        vector<Complex> test_state = {{0.4,-1.6},{1.2,0.7},{-0.1,0.6},{-1.3,0.4},{1.2,-1.3},{-1.2,1.7},{-3.1,0.6},{-0.3,0.2}};
//        vnormalize(test_state);
//        gState = test_state;
//        cout<<"initial state:"<<endl;
//        sparse_print(gState);
//        for(uint jj=0; jj<3; ++jj){
//            cout<<"draw C (qubits 0 and 1 involved)"<<endl;
//            gCi = draw_C();
//            cout<<"drawn "<<gCi<<", apply it"<<endl;
//            apply_C(gCi);
//            sparse_print(gState);
//        }
//        cout<<"measure qubit 1"<<endl;
//        uint ctest;
//        measure_qbit(gState, 1U, ctest);
//        sparse_print(gState);
//    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs_passed = (1./1000.)*std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cout<<"All [DONE] in "<<secs_passed<<" seconds"<<endl;

    return 0;
}
