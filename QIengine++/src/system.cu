#include "system.cuh"
#include "include/Rand.hpp"

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
  
    DEBUG_CALL(printf("after init_state()\n"));
    DEBUG_READ_STATE(state);


//    state.resize(Dim);
//    std::fill_n(state.begin(), state.size(), 0.0);
//    state[1].x = 1.0; //TWOSQINV; 
////    state[3] = -TWOSQINV; 
}


/* Quantum evolutor of the state */

void inversion(ComplexVec& state, const bmReg& q){
//    suqa::apply_mcx(state,{q[0],q[2]},{1U,0U},q[1]); 
}

void left_multiplication(ComplexVec& state, const bmReg& qr1, const bmReg& qr2){
    suqa::apply_cx(state, qr1[0], qr2[0]);
}

void self_plaquette(ComplexVec& state, const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
//    inversion(state, qr1);
    left_multiplication(state, qr1, qr0);
//    inversion(state, qr1);
//    inversion(state, qr2);
    left_multiplication(state, qr2, qr0);
//    inversion(state, qr2);
    left_multiplication(state, qr3, qr0);
}

void inverse_self_plaquette(ComplexVec& state, const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
//    inversion(state, qr3);
    left_multiplication(state, qr3, qr0);
//    inversion(state, qr3);
    left_multiplication(state, qr2, qr0);
    left_multiplication(state, qr1, qr0);
}

// void cphases(ComplexVec& state, uint qaux, uint q0b, double alpha1, double alpha2){
//     suqa::apply_cx(state, qaux, q0b);
//     suqa::apply_cu1(state, q0b, qaux, alpha1, 1U);
//     suqa::apply_cx(state, qaux, q0b);
//     suqa::apply_cu1(state, q0b, qaux, alpha2, 1U);
// }

void self_trace_operator(ComplexVec& state, const bmReg& qr, double th){
  suqa::apply_u1(state,qr[0],th); 
}

void fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    suqa::apply_h(state, qr[0]);
    
}


void inverse_fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    fourier_transf_z2(state,qr);
}

void momentum_phase(ComplexVec& state, const bmReg& qr, double th1){
    suqa::apply_u1(state, qr[0], th1);
    DEBUG_CALL(printf("\tafter suqa::apply_cu1(state, qr[0], th1, 0U)\n"));
    DEBUG_READ_STATE(state);
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

    DEBUG_CALL(if(n>0) printf("g_beta = %.16lg, dt = %.16lg, thetas: %.16lg %.16lg\n", g_beta, dt, theta1, theta));

    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink1, theta);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE(state);
        inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE(state);

        self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink2, theta);
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

        momentum_phase(state, bm_qlink0, theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink0, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink1, theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink1, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink2, theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink2, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink3, theta1);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink3, theta1, theta2)\n"));
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

const uint op_bits = 3; // 2^op_bits is the number of eigenvalues for the observable
const bmReg bm_op = bm_qlink1; // where the measure has to be taken
const std::vector<double> op_vals = {2.0,0.0,-2.0, 0.0,0.0,0.0,0.0,0.0}; // eigvals

 
// change basis to the observable basis somewhere in the system registers
void apply_measure_rotation(ComplexVec& state){
    self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
}

// inverse of the above function
void apply_measure_antirotation(ComplexVec& state){
    inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);

}

// map the classical measure recorded in creg_vals
// to the corresponding value of the observable;
// there is no need to change it
double get_meas_opvals(const uint& creg_vals){
    return op_vals[creg_vals];
}

// actually perform the measure
// there is no need to change it
double measure_X(ComplexVec& state, pcg& rgen){
    std::vector<uint> classics(op_bits);
    
    apply_measure_rotation(state);

    std::vector<double> rdoubs(op_bits);
    for(auto& el : rdoubs){
        el = rgen.doub();
    }
    suqa::measure_qbits(state, bm_op, classics, rdoubs);

    apply_measure_antirotation(state);

    uint meas = 0U;
    for(uint i=0; i<op_bits; ++i){
        meas |= (classics[i] << i);
    }

    return get_meas_opvals(meas);

}

/* Moves facilities */

std::vector<double> C_weigthsums = {1./3, 2./3, 1./3, 1.0,
				    1./3, 2./3, 1./3, 1.0,
				    1./3, 2./3, 1./3, 1.0,
				    1./3, 2./3, 1./3, 1.0,
				    2./5, 2./5};
std::vector<bmReg> link = {bm_qlink0, bm_qlink1, bm_qlink2, bm_qlink3};
std::vector<double> phases = {1./3, sqrt(2.)/3, -1./3, -sqrt(2.)/3,
				    sqrt(2.)/3, 1./3, -sqrt(2.)/3, -1./3,
				    -1./3, -sqrt(2.)/3, 1./3, sqrt(2.)/3,
				    -sqrt(2.)/3, -1./3, sqrt(2.)/3, 1./3};

void link_kinevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%4;
  fourier_transf_z2(state, link[link_index]);
  momentum_phase(state, link[link_index], phases[Ci]);
  inverse_fourier_transf_z2(state, link[link_index]);
  
}

void inverse_link_kinevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%4;
  inverse_fourier_transf_z2(state, link[link_index]);
  momentum_phase(state, link[link_index], -phases[Ci]);
  fourier_transf_z2(state, link[link_index]);
  
}




void apply_C(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
  uint s = (Ci<8U) ? 0 : Ci;
  switch(s){
        case 0U:
	  link_kinevolve(state,Ci);
            break;
        case 8U:
	  self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
            break;
        case 9U:
	  self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
            break;
        default:
            throw std::runtime_error("ERROR: wrong move selection");
    }
}

void apply_C_inverse(ComplexVec& state, const bmReg& bm_states, const uint &Ci){
  switch(Ci){
        case 0U:
	  inverse_link_kinevolve(state,Ci);
            break;
        case 8U:
	  inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
            break;
        case 9U:
	  inverse_self_plaquette(state, bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
            break;
        default:
            throw std::runtime_error("ERROR: wrong move selection");
    }
}
    


std::vector<double> get_C_weigthsums(){ return C_weigthsums; }
