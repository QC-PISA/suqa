#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include "suqa.cuh"


/* d4 gauge theory - two plaquettes
 
   link state 3 qubits
   system state: 4 links -> 12 qubits


 */

typedef std::vector<uint> bmReg;

__global__ void initialize_state(double *state_re, double *state_im, uint len){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x*blockDim.x;
    }
    if(blockIdx.x*blockDim.x+threadIdx.x==1){
        state_re[0] = 1.0;
        state_im[0] = 0.0;
    }
}

const bmReg bm_qlink0 =  {0,  1, 2};
const bmReg bm_qlink1 =  {3,  4, 5};
const bmReg bm_qlink2 =  {6,  7, 8};
const bmReg bm_qlink3 =  {9, 10, 11};
const bmReg bm_qaux   = {12, 13, 14};

double g_beta;
double theta1, theta2, theta;

__inline__ double f1(double b){
    return log((3+cosh(2.*b))/(2*sinh(b)*sinh(b)));
}

__inline__ double f2(double b){
    return -log(tanh(b));
}

void init_state(ComplexVec& state, uint Dim, double gg_beta){

    g_beta = gg_beta;

    if(state.size()!=Dim)
        throw std::runtime_error("ERROR: init_state() failed");
    
    // zeroes entries and set state to all the computational element |000...00>
    initialize_state<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(state.data_re, state.data_im,Dim);
    cudaDeviceSynchronize();


    //TODO: optimize
    suqa::apply_h(state, bm_qlink0[0]);
    suqa::apply_cx(state, bm_qlink0[0], bm_qlink3[0]);
    suqa::apply_h(state, bm_qlink0[1]);
    suqa::apply_cx(state, bm_qlink0[1], bm_qlink3[1]);
    suqa::apply_h(state, bm_qlink0[2]);
    suqa::apply_cx(state, bm_qlink0[2], bm_qlink3[2]);
    suqa::apply_mcx(state, {bm_qlink3[0], bm_qlink3[2]}, {0U,1U}, bm_qlink3[1]);


//    state.resize(Dim);
//    std::fill_n(state.begin(), state.size(), 0.0);
//    state[1].x = 1.0; //TWOSQINV; 
////    state[3] = -TWOSQINV; 
}

/* Hamiltonian
 *
 * H = E = 0, 1/2, 1/sqrt(2), 3/4
 *
 */


__global__ 
void kernel_cevolution(double *const state_re, double *const state_im, uint len, uint mask, uint cmask, uint qstate0, uint qstate1, Complex ph1, Complex ph2, Complex ph3){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    double tmpval;
    while(i_0<len){
        if((i_0 & mask) == cmask){
      
            uint i_1 = i_0 | (1U << qstate0);
            uint i_2 = i_0 | (1U << qstate1);
            uint i_3 = i_1 | i_2;
            
//            state[i_0] = a_0;
            tmpval = state_re[i_1];
            state_re[i_1] = state_re[i_1]*ph1.x - state_im[i_1]*ph1.y;
            state_im[i_1] = state_im[i_1]*ph1.x + tmpval*ph1.y;
            tmpval = state_re[i_2];
            state_re[i_2] = state_re[i_2]*ph2.x - state_im[i_2]*ph2.y;
            state_im[i_2] = state_im[i_2]*ph2.x + tmpval*ph2.y;
            tmpval = state_re[i_3];
            state_re[i_3] = state_re[i_3]*ph3.x - state_im[i_3]*ph3.y;
            state_im[i_3] = state_im[i_3]*ph3.x + tmpval*ph3.y;
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

// void cevolution(ComplexVec& state, const double& t, const int& n, const uint& q_control, const std::vector<uint>& qstate){
// 
//      (void)n; // Trotter not needed
//      double dt = t;
//  
// 
//     if(qstate.size()!=2)
//         throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");
// 
//     uint cmask = (1U << q_control);
// 	uint mask = cmask;
//     for(const auto& qs : qstate){
//         mask |= (1U << qs);
//     }
//     //TODO: implement device code
//     kernel_cevolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, cmask, qstate[0], qstate[1], expi(-dt*eig1), expi(-dt*eig2), expi(-dt*eig3));
// }

/* Quantum evolutor of the state */

void inversion(ComplexVec& state, const bmReg& q){
    suqa::apply_mcx(state,{q[0],q[2]},{1U,0U},q[1]); 
}

void left_multiplication(ComplexVec& state, const bmReg& qr1, const bmReg& qr2){
    suqa::apply_cx(state, qr1[1], qr2[1]);
    suqa::apply_mcx(state, {qr1[0], qr2[0]}, qr2[1]);
    suqa::apply_cx(state, qr1[0], qr2[0]);
    suqa::apply_mcx(state, {qr1[0], qr2[2]}, qr2[1]);
    suqa::apply_cx(state, qr1[2], qr2[2]);
}

void self_plaquette(ComplexVec& state, const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    inversion(state, qr1);
    left_multiplication(state, qr1, qr0);
    inversion(state, qr1);
    inversion(state, qr2);
    left_multiplication(state, qr2, qr0);
    inversion(state, qr2);
    left_multiplication(state, qr3, qr0);
}

void inverse_self_plaquette(ComplexVec& state, const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    inversion(state, qr3);
    left_multiplication(state, qr3, qr0);
    inversion(state, qr3);
    left_multiplication(state, qr2, qr0);
    left_multiplication(state, qr1, qr0);
}

void cphases(ComplexVec& state, const uint& qaux, const uint& q0b, double alpha1, double alpha2){
    suqa::apply_mcu1(state, q0b, qaux, alpha1, 0U);
    suqa::apply_mcu1(state, q0b, qaux, alpha2, 1U);
}

void self_trace_operator(ComplexVec& state, const bmReg& qr, const uint& qaux){
    //TODO: optimize
    suqa::apply_mcx(state, {qr[0],qr[2]}, {0U,0U}, qaux); 
    cphases(state, qaux, qr[1],theta, -theta);
    suqa::apply_mcx(state, {qr[0],qr[2]}, {0U,0U}, qaux); 
}

void cevolution(ComplexVec& state, const double& t, const int& n, const uint& q_control, const std::vector<uint>& qstate){

    if(qstate.size()!=15)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

//    uint cmask = (1U << q_control);
//	uint mask = cmask;
//    for(const auto& qs : qstate){
//        mask |= (1U << qs);
//    }

    const double dt = t/(double)n;

    theta1 = dt*f1(g_beta);
    theta2 = dt*f2(g_beta);
    theta = 2*dt*g_beta;

    //TODO: continue
    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        self_trace_operator(state, bm_qlink1, bm_qaux[0]);
        inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);

        self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        self_trace_operator(state, bm_qlink2, bm_qaux[0]);
        inverse_self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
    }

//    d4.fourier_trans(qc, q0)
//    d4.fourier_trans(qc, q1)
//    d4.fourier_trans(qc, q2)
//    d4.fourier_trans(qc, q3)
//
//    d4.momentum_phase(qc, q0, q_aux0, q_aux1, q_aux2, theta1, theta2)
//    d4.momentum_phase(qc, q1, q_aux0, q_aux1, q_aux2, theta1, theta2)
//    d4.momentum_phase(qc, q2, q_aux0, q_aux1, q_aux2, theta1, theta2)
//    d4.momentum_phase(qc, q3, q_aux0, q_aux1, q_aux2, theta1, theta2)
// 
//    d4.inverse_fourier_trans(qc, q0)
//    d4.inverse_fourier_trans(qc, q1)
//    d4.inverse_fourier_trans(qc, q2)
//    d4.inverse_fourier_trans(qc, q3)

//    kernel_cevolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, cmask, qstate[0], qstate[1], expi(-dt*eig1), expi(-dt*eig2), expi(-dt*eig3));
}

/* Hamiltonian
 *
 * H = 1/4 (1 + X1 X0 + X2 X0 + X2 X1)
 *
 */

//void cevolution(std::vector<std::complex<double>>& state, const double& t, const int& n, const uint& q_control, const std::vector<uint>& qstate){
//
//    (void)n; // Trotter not needed
//    double dt = t;
//
//    if(qstate.size()!=3)
//        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");
//     uint cmask = (1U << q_control);
//    uint mask = cmask; // (1U << qstate[0]) | (1U << qstate[0])
//     for(const auto& qs : qstate){
//         mask |= (1U << qs);
//     }
//
//    for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
//         if((i_0 & mask) == cmask){
//       
//             uint i_1 = i_0 | (1U << qstate[0]);
//             uint i_2 = i_0 | (1U << qstate[1]);
//             uint i_3 = i_1 | i_2;
//             uint i_4 = i_0 | (1U << qstate[2]);
//             uint i_5 = i_4 | i_1;
//             uint i_6 = i_4 | i_2;
//             uint i_7 = i_4 | i_3;
//
//
//             Complex a_0 = state[i_0];
//             Complex a_1 = state[i_1];
//             Complex a_2 = state[i_2];
//             Complex a_3 = state[i_3];
//             Complex a_4 = state[i_4];
//             Complex a_5 = state[i_5];
//             Complex a_6 = state[i_6];
//             Complex a_7 = state[i_7];
//
//             double dtp = dt/4.; 
//             // apply 1/.4 (Id +X2 X1)
//             state[i_0] = exp(-dtp*iu)*(cos(dtp)*a_0 -sin(dtp)*iu*a_6);
//             state[i_1] = exp(-dtp*iu)*(cos(dtp)*a_1 -sin(dtp)*iu*a_7);
//             state[i_2] = exp(-dtp*iu)*(cos(dtp)*a_2 -sin(dtp)*iu*a_4);
//             state[i_3] = exp(-dtp*iu)*(cos(dtp)*a_3 -sin(dtp)*iu*a_5);
//             state[i_4] = exp(-dtp*iu)*(cos(dtp)*a_4 -sin(dtp)*iu*a_2);
//             state[i_5] = exp(-dtp*iu)*(cos(dtp)*a_5 -sin(dtp)*iu*a_3);
//             state[i_6] = exp(-dtp*iu)*(cos(dtp)*a_6 -sin(dtp)*iu*a_0);
//             state[i_7] = exp(-dtp*iu)*(cos(dtp)*a_7 -sin(dtp)*iu*a_1);
//
//             a_0 = state[i_0];
//             a_1 = state[i_1];
//             a_2 = state[i_2];
//             a_3 = state[i_3];
//             a_4 = state[i_4];
//             a_5 = state[i_5];
//             a_6 = state[i_6];
//             a_7 = state[i_7];
//
//             // apply 1/.4 (X2 X0)
//             state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_5);
//             state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_4);
//             state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_7);
//             state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_6);
//             state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_1);
//             state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_0);
//             state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_3);
//             state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_2);
//
//             a_0 = state[i_0];
//             a_1 = state[i_1];
//             a_2 = state[i_2];
//             a_3 = state[i_3];
//             a_4 = state[i_4];
//             a_5 = state[i_5];
//             a_6 = state[i_6];
//             a_7 = state[i_7];
//
//             // apply 1/.4 (X1 X0)
//             state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_3);
//             state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_2);
//             state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_1);
//             state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_0);
//             state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_7);
//             state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_6);
//             state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_5);
//             state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_4);
//         }
//    }
//} 

/* Measure facilities */
uint state_levels;
std::vector<double> meas_opvals;
std::vector<std::vector<Complex>> SXmat;
std::vector<uint> iss;//(state_levels);
std::vector<Complex> ass;//(state_levels);
uint meas_mask;
std::vector<uint> meas_mask_combs;


void fill_meas_cache(const bmReg& bm_states, const std::string opstem){
    state_levels = (1U << bm_states.size());

    iss.resize(state_levels);
    ass.resize(state_levels);

    meas_opvals.resize(state_levels);
    SXmat.resize(state_levels,std::vector<Complex>(state_levels));

    FILE * fil_re = fopen((opstem+"_vecs_re").c_str(),"r"); 
    FILE * fil_im = fopen((opstem+"_vecs_im").c_str(),"r"); 
    FILE * fil_vals = fopen((opstem+"_vals").c_str(),"r"); 
    double tmp_re,tmp_im;
    int fscanf_items=1;
    for(uint i=0; i<state_levels; ++i){
        fscanf_items=fscanf(fil_vals, "%lg",&meas_opvals[i]);
        for(uint j=0; j<state_levels; ++j){
            fscanf_items*=fscanf(fil_re, "%lg",&tmp_re);
            fscanf_items*=fscanf(fil_im, "%lg",&tmp_im);
            SXmat[i][j].x = tmp_re;
            SXmat[i][j].y = tmp_im;
        }
        fscanf_items*=1-fscanf(fil_re, "\n");
        fscanf_items*=1-fscanf(fil_im, "\n");
    }
    if(fscanf_items!=1){
        std::cout<<fscanf_items<<std::endl;
        throw std::runtime_error("ERROR: while reading Xmatstem files");
    }

    fclose(fil_vals);
    fclose(fil_re);
    fclose(fil_im);

    meas_mask = 0U;
    for(const auto& bm : bm_states){
        meas_mask |= (1U<<bm);
    }
    meas_mask_combs.resize(state_levels,0);
    for(uint lvl=0; lvl<state_levels; ++lvl){
        for(uint bmi=0; bmi<bm_states.size(); ++bmi){
            meas_mask_combs[lvl] |= ((lvl>>bmi & 1U) << bm_states[bmi]);
        }
    }
}

double get_meas_opvals(const uint& creg_vals){
    return meas_opvals[creg_vals];
}
 
void apply_measure_rotation(ComplexVec& state){

//	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
//        if((i_0 & meas_mask) == 0U){
//      
//            for(uint lvl=0; lvl<state_levels; ++lvl){
//                iss[lvl] = i_0 | meas_mask_combs[lvl];
//                ass[lvl] = state[iss[lvl]];
//            }
//
//            for(uint r=0; r<state_levels; ++r){
//                state[iss[r]].x=0.0;
//                state[iss[r]].y=0.0;
//                for(uint c=0; c<state_levels; ++c){
//                     state[iss[r]] += SXmat[r][c]*ass[c];
//                }
//            }
//        }
//    }
}

void apply_measure_antirotation(ComplexVec& state){
//
//	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
//        if((i_0 & meas_mask) == 0U){
//      
//            for(uint lvl=0; lvl<state_levels; ++lvl){
//                iss[lvl] = i_0 | meas_mask_combs[lvl];
//                ass[lvl] = state[iss[lvl]];
//            }
//
//            for(uint r=0; r<state_levels; ++r){
//                state[iss[r]]=0.0;
//                for(uint c=0; c<state_levels; ++c){
//                     state[iss[r]] += conj(SXmat[c][r])*ass[c];
//                }
//            }
//        }
//    }
}

/* Metropolis update step */
//std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};
//void qi_h(std::vector<Complex>& state, const uint& q);
//void apply_C(std::vector<Complex>& state, const std::vector<uint>& bm_states, const uint &Ci){
//    if(Ci==0U){
//        qi_h(state,bm_states[0]);
//    }else if(Ci==1U){
//        qi_h(state,bm_states[1]);
//    }else if(Ci==2U){
//        qi_h(state,bm_states[2]);
//    }else{
//        throw std::runtime_error("Error!");
//    }
//}

std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};

void apply_C(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
    if(Ci==0U){
        suqa::apply_cx(state,bm_states[1], 0, bm_states[0]);
    }else if(Ci==1U){
        suqa::apply_swap(state,bm_states[1],bm_states[0]);
    }else if(Ci==2U){
        suqa::apply_x(state,bm_states);
    }else{
        throw "Error!";
    }
}

void apply_C_inverse(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
    apply_C(state, bm_states, Ci);
}

std::vector<double> get_C_weigthsums(){ return C_weigthsums; }
