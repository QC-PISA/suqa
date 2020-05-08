#include "system.cuh"
#include "Rand.hpp"



//TODO: make the number of "state" qubits determined at compilation time in system.cuh
double g_beta;


__global__ void initialize_state(double *state_re, double *state_im, uint len, int j){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x*blockDim.x;
    }
    if(blockIdx.x*blockDim.x+threadIdx.x==0){
      switch(j){
      case 0:
        state_re[0] = 1.0;
        state_im[0] = 0.0;
      	break;
      default:
	// state_re[j] = 1.0;
	// state_im[j]= 0;
	// state_re[j+9-(2*(j%2))] = sqrt(0.5);
	// state_im[j+9-(2*(j%2))] = 0.0;
	break;
      }
    }
}

__inline__ double fp(double b){
  return log(exp(2*b)+1);
}

__inline__ double fm(double b){
  if (b<0) std::runtime_error("ERROR: fm(b) failed because of b<0");
  return log(exp(2*b)-1);
}


void init_state(ComplexVec& state, uint Dim, uint j=0){

    if(state.size()!=Dim)
	throw std::runtime_error("ERROR: init_state() failed");

    if(j>=8)
      throw std::runtime_error("ERROR: attempt to initialize with more than 8 basis vectors init_state()");

    // zeroes entries and set state to all the computational element |000...00>
    initialize_state<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, Dim, j);
    cudaDeviceSynchronize();

    if(j==0){
      suqa::apply_h(state, bm_qlink0[0]);
     suqa::apply_h(state, bm_qlink1[0]);
      suqa::apply_h(state, bm_qlink2[0]);
      // suqa::apply_cx(state,bm_qlink0[0],bm_qlink1[0]);
      // suqa::apply_cx(state,bm_qlink0[0],bm_qlink2[0]);
      // suqa::apply_z(state, bm_qlink0[0]);
   }
    
    DEBUG_CALL(printf("after init_state()\n"));
    DEBUG_READ_STATE(state);
}


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
  suqa::apply_u1(state, qr[0], 0U, th);
  DEBUG_CALL(printf("\tafter self_trace_operator(state, qr, th1, th2)\n"));
  DEBUG_READ_STATE(state);
}

void fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    suqa::apply_h(state, qr[0]);
    DEBUG_CALL(printf("\tafter fourier_transf_z2(state, qr, th1, th2)\n"));
    DEBUG_READ_STATE(state);    
}


void inverse_fourier_transf_z2(ComplexVec& state, const bmReg& qr){
    fourier_transf_z2(state,qr);
    DEBUG_CALL(printf("\tafter inverse_fourier_transf_z2(state, qr, th1, th2)\n"));
    DEBUG_READ_STATE(state);    
}

void momentum_phase(ComplexVec& state, const bmReg& qr, double th1, double th2){
    suqa::apply_u1(state, qr[0],th2);
    suqa::apply_u1(state, qr[0],0U, th1);
    DEBUG_CALL(printf("\tafter momentum_phase(state, qr, th1, th2)\n"));
    DEBUG_READ_STATE(state);
}

void evolution(ComplexVec& state, const double& t, const int& n){

    const double dt = -t/(double)n;

    const double theta1 = dt*fp(g_beta);
    const double theta2 = dt*fm(g_beta);
    const double theta = 4*dt*g_beta;

    DEBUG_CALL(if(n>0) printf("g_beta = %.16lg, dt = %.16lg, thetas: %.16lg %.16lg\n", g_beta, dt, theta1, theta));

    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink1, theta*0.5);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE(state);
        inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
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
        // fourier_transf_z2(state, bm_qlink3);
        // DEBUG_CALL(printf("after fourier_transf_z2(state, bm_qlink3)\n"));
        // DEBUG_READ_STATE(state);

        momentum_phase(state, bm_qlink0, 2*theta1, 2*theta2);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink0, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink1, theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink1, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        momentum_phase(state, bm_qlink2, theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(state, bm_qlink2, theta1, theta2)\n"));
        DEBUG_READ_STATE(state);
        // momentum_phase(state, bm_qlink3, theta1, theta2);
        // DEBUG_CALL(printf("after momentum_phase(state, bm_qlink3, theta1, theta2)\n"));
        // DEBUG_READ_STATE(state);
	

        // inverse_fourier_transf_z2(state, bm_qlink3);
        // DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink3)\n"));
        // DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink2);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink2)\n"));
        DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink1);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink1)\n"));
        DEBUG_READ_STATE(state);
        inverse_fourier_transf_z2(state, bm_qlink0);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(state, bm_qlink0)\n"));
        DEBUG_READ_STATE(state);

        self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE(state);
        self_trace_operator(state, bm_qlink1, theta*0.5);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE(state);
        inverse_self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE(state);

    }
}


/* Measure facilities */

const uint op_bits = 1; // 2^op_bits is the number of eigenvalues for the observable
const bmReg bm_op = bm_qlink1; // where the measure has to be taken
const std::vector<double> op_vals = {2.0,0.0}; // eigvals

 
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

std::vector<double> C_weigthsums = {1./11, 2./11, 3./11, 4./11, 5./11, 6./11, 7./11, 8./11, 9./11, 10./11, 1.};
/*
std::vector<double> C_weigthsums = {1./24, 2./24, 3./24, 4./24, //0<=Ci<=3
				    5./24, 6./24, 7./24, 8./24, //4<=Ci<=7
				    9./24, 10./24, 11./24, 12./24, //8<=Ci<=11
				    13./24, 14./24, 15./24, 16./24, //12<=Ci<=15
				    17./24, 18./24, 19./24, 20./24, //16<=Ci<=19
				    21./24, 22./24, 23./24, 1.}; //20<=Ci<=23

std::vector<bmReg> link = {bm_qlink0, bm_qlink1, bm_qlink2, bm_qlink3};

std::vector<double> phases = {1./3, sqrt(2.)/3};

void link_kinevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%4;
  fourier_transf_z2(state, link[link_index]);
  momentum_phase(state, link[link_index],fp(phases[(Ci/4)%2])*(((int)Ci/8)*2-1),fm(phases[(Ci/4)%2])*(((int)Ci/8)*2-1)); //0-1 1-1 2-1 3-1 0-2 1-2 2-2 3-2 01 11 21 31 02 12 22 32
  inverse_fourier_transf_z2(state, link[link_index]);
  
}

void inverse_link_kinevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%4;
  inverse_fourier_transf_z2(state, link[link_index]);
  momentum_phase(state, link[link_index], -fp(phases[(Ci/4)%2])*(((int)Ci/8)*2-1), -fm(phases[(Ci/4)%2])*(((int)Ci/8)*2-1));
  fourier_transf_z2(state, link[link_index]);
  
}

void link_plaqevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%2;  //(Ci%2)+1 (Ci%2)*3  2-(Ci%2)     (Ci%2)*3
  self_plaquette(state, link[link_index+1], link[link_index*3], link[2-link_index], link[link_index*3]);
  self_trace_operator(state, link[1], phases[(Ci/2)%2]*(((int)Ci/20)*2-1));// p1-1 p2-1 p1-2 p2-2 p11 p21 p12 p22
  inverse_self_plaquette(state, link[link_index+1], link[link_index*3], link[2-link_index], link[link_index*3]);  
}

void inverse_link_plaqevolve(ComplexVec& state, const uint&Ci){
  int link_index=Ci%2;  //(Ci%2)+1 (Ci%2)*3  2-(Ci%2)     (Ci%2)*3
  inverse_self_plaquette(state, link[link_index+1], link[link_index*3], link[2-link_index], link[link_index*3]);
  self_trace_operator(state, link[1], -phases[(Ci/2)%2]*(((int)Ci/20)*2-1));// p1-1 p2-1 p1-2 p2-2 p11 p21 p12 p22
  self_plaquette(state, link[link_index+1], link[link_index*3], link[2-link_index], link[link_index*3]);  
}






void apply_C(ComplexVec& state, const uint &Ci){
  uint s = (Ci<16U) ? 0 : 1;
  switch(s){
        case 0U:
	  link_kinevolve(state,Ci);
            break;
        case 1U:
	 link_plaqevolve(state,Ci);
            break;
        default:
            throw std::runtime_error("ERROR: wrong move selection");
    }
}

void apply_C_inverse(ComplexVec& state, const uint &Ci){
  uint s = (Ci<16U) ? 0 : 1;
  switch(s){
        case 0U:
	  inverse_link_kinevolve(state,Ci);
            break;
        case 1U:
	  inverse_link_plaqevolve(state,Ci);
            break;
        default:
            throw std::runtime_error("ERROR: wrong move selection");
    }
}
    
*/

void apply_C(ComplexVec& state, const uint &Ci){
  switch(Ci){
  case 0U:
    suqa::apply_z(state, bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_z(state, bm_qlink1[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 1U:
    suqa::apply_z(state, bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_z(state, bm_qlink2[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 2U:
    suqa::apply_z(state, bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_z(state, bm_qlink0[0])\n"));
   DEBUG_READ_STATE(state);
    // suqa::apply_z(state, bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_z(state, bm_qlink3[0])\n"));
    // DEBUG_READ_STATE(state);
    break;
  case 3U:
    suqa::apply_y(state, bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink0[0])\n"));
    DEBUG_READ_STATE(state);
   // suqa::apply_y(state, bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_y(state, bm_qlink3[0])\n"));
    // DEBUG_READ_STATE(state);
   break;
  case 4U:
    suqa::apply_y(state, bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink1[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 5U:
    suqa::apply_y(state, bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink2[0])\n"));
   DEBUG_READ_STATE(state);
    break;
  case 6U:
    suqa::apply_x(state, bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink0[0])\n"));
    DEBUG_READ_STATE(state);
    // suqa::apply_y(state, bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_y(state, bm_qlink3[0])\n"));
    // DEBUG_READ_STATE(state);
    break;
  case 7U:
    suqa::apply_x(state, bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink1[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 8U:
    suqa::apply_x(state, bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink2[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 9U:
    suqa::apply_h(state, bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink2[0])\n"));
   DEBUG_READ_STATE(state);
    break;
  case 10U:
    suqa::apply_h(state, bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink2[0])\n"));
    DEBUG_READ_STATE(state);
    break;
  case 11U:
    suqa::apply_h(state, bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(state, bm_qlink2[0])\n"));
    DEBUG_READ_STATE(state);
    break;
    
  default:
    throw std::runtime_error("ERROR: wrong move selection");
  }
}


void apply_C_inverse(ComplexVec& state, const uint &Ci){
  apply_C(state,Ci);
}

std::vector<double> get_C_weigthsums(){ return C_weigthsums; }

