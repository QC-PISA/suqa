#include "system.cuh"

/* d4 gauge theory - two plaquettes
 
   link state 3 qubits
   system state: 4 links -> 12 qubits
   +1 ancillary qubit

 */

//TODO: make the number of "state" qubits determined at compilation time in system.cuh
double g_beta;


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


__inline__ double f1(double b){
  return log(tanh(b));
}


void init_state(ComplexVec& state, uint Dim){

    if(state.size()!=Dim)
        throw std::runtime_error("ERROR: init_state() failed");
    
    // zeroes entries and set state to all the computational element |000...00>
    initialize_state<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(state.data_re, state.data_im,Dim);
    cudaDeviceSynchronize();


    suqa::apply_h(state, bm_qlink0[0]);
    suqa::apply_cx(state, bm_qlink0[0], bm_qlink3[0]);
  

//    state.resize(Dim);
//    std::fill_n(state.begin(), state.size(), 0.0);
//    state[1].x = 1.0; //TWOSQINV; 
////    state[3] = -TWOSQINV; 
}


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

void cphases(ComplexVec& state, uint qaux, uint q0b, double alpha1, double alpha2){
    suqa::apply_cx(state, qaux, q0b);
    suqa::apply_cu1(state, q0b, qaux, alpha1, 1U);
    suqa::apply_cx(state, qaux, q0b);
    suqa::apply_cu1(state, q0b, qaux, alpha2, 1U);
}

void self_trace_operator(ComplexVec& state, const bmReg& qr, const uint& qaux, double th){
    suqa::apply_mcx(state, {qr[0],qr[2]}, {0U,0U}, qaux); 
    cphases(state, qaux, qr[1], th, -th);
    suqa::apply_mcx(state, {qr[0],qr[2]}, {0U,0U}, qaux); 
}

void fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    suqa::apply_h(state, qr[0]);
    
}


void inverse_fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    fourier_transf_z2(state,qr);
}

void momentum_phase(ComplexVec& state, const bmReg& qr, const uint& qaux, double th1){
    suqa::apply_cu1(state, qaux, qr[0], th1);
    DEBUG_CALL(printf("\tafter suqa::apply_cu1(state, qaux, qr[0], th1, 0U)\n"));
    DEBUG_READ_STATE(state);
}

void cevolution(ComplexVec& state, const double& t, const int& n, const uint& q_control, const bmReg& qstate){
//    if(qstate.size()!=13)
//        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

}

void evolution(ComplexVec& state, const double& t, const int& n){


//    uint cmask = (1U << q_control);
//	uint mask = cmask;
//    for(const auto& qs : qstate){
//        mask |= (1U << qs);
//    }

    const double dt = t/(double)n;

    const double theta1 = dt*f1(g_beta);
    const double theta = 2*dt*g_beta;
//    printf("g_beta = %.16lg, dt = %.16lg, thetas: %.16lg %.16lg %.16lg\n", g_beta, dt, theta1, theta2, theta);

    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink1, bm_qaux[0], theta);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE(state);
        inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE(state);

        self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink2, bm_qaux[0], theta);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE(state);
        inverse_self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE(state);

        fourier_transf_z2(state, bm_qlink0);
        DEBUG_CALL(printf("after fourier_transf_z2(state, bm_qlink0)\n"));
        DEBUG_READ_STATE(state);
        fourier_transf_z2(state, bm_qlink1);
        DEBUG_CALL(printf("after fourier_transf_z2(state, bm_qlink1)\n"));
        DEBUG_READ_STATE(state);
        fourier_transf_z2(state, bm_qlink2);
        DEBUG_CALL(printf("after fourier_transf_z2(state, bm_qlink2)\n"));
        DEBUG_READ_STATE(state);
        fourier_transf_z2(state, bm_qlink3);
        DEBUG_CALL(printf("after fourier_transf_z2(state, bm_qlink3)\n"));
        DEBUG_READ_STATE(state);

        momentum_phase(state, bm_qlink0, bm_qaux[0], theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink0, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink1, bm_qaux[0], theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink1, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink2, bm_qaux[0], theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink2, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink3, bm_qaux[0], theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink3, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
	

        inverse_fourier_transf_z2(state, bm_qlink3);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink3)\n"));
        DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink2);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink2)\n"));
        DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink1);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink1)\n"));
        DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink0);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink0)\n"));
        DEBUG_READ_STATE(state);
    }
}


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


//TODO: write moves
std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};

void apply_C(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
    switch(Ci){
        case 0U:
            suqa::apply_cx(state,bm_states[1], 0, bm_states[0]);
            break;
        case 1U:
            suqa::apply_swap(state,bm_states[1],bm_states[0]);
            break;
        case 2U:
            suqa::apply_x(state,bm_states);
            break;
        default:
            throw std::runtime_error("ERROR: wrong move selection");
    }
}

void apply_C_inverse(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
    apply_C(state, bm_states, Ci);
}

std::vector<double> get_C_weigthsums(){ return C_weigthsums; }
