#include "system.cuh"

/* z2 gauge theory - two plaquettes
 
    link state 1 qubit
    system state: after gauge fixing 3 links -> 3 qubits 

    .   .   .
    1   2
    o 0 o I .

    //TODO: adapt from z2
//XXX    operation table for the D4 group:
//XXX
//XXX    [abc -> a*4+b*2+c]
//XXX
//XXX    representation: ρ_{abc} = (-1)^b [[0,1],[1,0]]^a [[i, 0],[0,-i]]^c
//XXX
//XXX    ρ_{a'b'c'}ρ_{abc} = (-1)^{b'+b} 
//XXX
//XXX     
//XXX          0   1   2   3   4   5   6   7
//XXX        ________________________________
//XXX       |
//XXX    0  |  0   1   2   3   4   5   6   7
//XXX       |
//XXX    1  |  1   2   3   0   7   4   5   6 
//XXX       | 
//XXX    2  |  2   3   0   1   6   7   4   5
//XXX       |
//XXX    3  |  3   0   1   2   5   6   7   4
//XXX       |
//XXX    4  |  4   7   6   5   0   1   2   3
//XXX       |
//XXX    5  |  5   4   7   6   1   0   1   2
//XXX       |
//XXX    6  |  6   5   4   7   2   1   0   1
//XXX       |
//XXX    7  |  7   6   5   4   3   2   1   0

 */



double g_beta;

__inline__ double fp(double b){
  return log(exp(2*b)+1);
}

__inline__ double fm(double b){
  if (b<0) std::runtime_error("ERROR: fm(b) failed because of b<0");
  return log(exp(2*b)-1);
}

void init_state(){
    suqa::init_state();

    suqa::apply_h(bm_qlink0[0]);
    suqa::apply_h(bm_qlink1[0]);
    suqa::apply_h(bm_qlink2[0]);
//    suqa::apply_h(bm_qlink0[0]);
//    suqa::apply_cx(bm_qlink0[0], bm_qlink3[0]);
//    suqa::apply_h(bm_qlink0[1]);
//    suqa::apply_cx(bm_qlink0[1], bm_qlink3[1]);
//    suqa::apply_h(bm_qlink0[2]);
//    suqa::apply_cx(bm_qlink0[2], bm_qlink3[2]);
//    suqa::apply_mcx({bm_qlink3[0], bm_qlink3[2]}, {0U,1U}, bm_qlink3[1]);

// automatically gauge-invariant initialization
/*    .   .   .
 *    1   2
 *    o 0 o I .
 *   g1  g2
 *  
 *  U0 -> g2 U0 g1' -> g
 *  U1 -> g1 U1 g1' -> I
 *  U2 -> g2 U2 g2' -> I
 *  U3 -> g1 U3 g2' -> g'
 *
 *  g=g2 g1', sum over g
 *
 *  GF: g2=g1 U3
 *  
 *  U0 -> U0n = g1 U3 U0 g1'     -> I
 *  U1 -> U1n = g1 U1 g1'        -> I
 *  U2 -> U2n = g1 U3 U2 U3' g1' -> I
 *  U3 -> U3n = I                -> I
 *
 */

}


/* Quantum evolutor of the state */

//void inversion(const bmReg& q){
////    suqa::apply_mcx({q[0],q[2]},{1U,0U},q[1]); 
//}

void left_multiplication(const bmReg& qr1, const bmReg& qr2){
    // applies group element from register qr1 to register qr2
    // |...,U_{qr1},...,U_{qr2},...> -> |...,U_{qr1},...,U_{qr1}U_{qr2},...>
    suqa::apply_cx(qr1[0], qr2[0]);
//    suqa::apply_cx(qr1[1], qr2[1]);
//    suqa::apply_mcx({qr1[0], qr2[0]}, qr2[1]);
//    suqa::apply_cx(qr1[0], qr2[0]);
//    suqa::apply_mcx({qr1[0], qr2[2]}, qr2[1]);
//    suqa::apply_cx(qr1[2], qr2[2]);
}

void self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    // applies the following operation in the computational basis
    // |...,U_{qr0}...,U_{qr1}...,U_{qr2},...,U_{qr3},...> -> 
    //                      .
    //                      .
    //                      V
    // |...,U_{qr3}U'_{qr2}U'_{qr1}U_{qr0}...,U_{qr1}...,U_{qr2},...,U_{qr3},...>
//    inversion(qr1);
    left_multiplication(qr1, qr0);
//    inversion(qr1);
//    inversion(qr2);
    left_multiplication(qr2, qr0);
//    inversion(qr2);
    left_multiplication(qr3, qr0);
}

void inverse_self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    // inverse operation of self_plaquette
//    inversion(qr3);
    left_multiplication(qr3, qr0);
//    inversion(qr3);
    left_multiplication(qr2, qr0);
    left_multiplication(qr1, qr0);
}


//void self_plaquette1(){
//    self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
//}
//void inverse_self_plaquette1(){
//    inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
//}
//
//void self_plaquette2(){
//    inversion(bm_qlink1);
//    left_multiplication(bm_qlink1,bm_qlink2);
//    inversion(bm_qlink1);
//}
//void inverse_self_plaquette2(){
//}

//void cphases(uint qaux, uint q0b, double alpha1, double alpha2){
//    // eigenvalues of the trace operator
//    suqa::apply_cx(qaux, q0b);
//    suqa::apply_cu1(q0b, qaux, alpha1, 1U);
//    suqa::apply_cx(qaux, q0b);
//    suqa::apply_cu1(q0b, qaux, alpha2, 1U);
//}

void self_trace_operator(const bmReg& qr, double th){
  suqa::apply_u1(qr[0], 0U, th);
  DEBUG_CALL(printf("\tafter self_trace_operator(qr, th1, th2)\n"));
  DEBUG_READ_STATE();
}

void fourier_transf_z2(const bmReg& qr){
    suqa::apply_h(qr[0]);
    DEBUG_CALL(printf("\tafter fourier_transf_z2(qr, th1, th2)\n"));
    DEBUG_READ_STATE();    
}


void inverse_fourier_transf_z2(const bmReg& qr){
    fourier_transf_z2(qr);
    DEBUG_CALL(printf("\tafter inverse_fourier_transf_z2(qr, th1, th2)\n"));
    DEBUG_READ_STATE();    
}

void momentum_phase(const bmReg& qr, double th1, double th2){
    suqa::apply_u1(qr[0],th2);
    suqa::apply_u1(qr[0],0U, th1);
    DEBUG_CALL(printf("\tafter momentum_phase(qr, th1, th2)\n"));
    DEBUG_READ_STATE();
}

void evolution(const double& t, const int& n){

    const double dt = -t/(double)n;

    const double theta1 = dt*fp(g_beta);
    const double theta2 = dt*fm(g_beta);
    const double theta = 4*dt*g_beta;

    DEBUG_CALL(if(n>0) printf("g_beta = %.16lg, dt = %.16lg, thetas: %.16lg %.16lg\n", g_beta, dt, theta1, theta));

    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE();
        self_trace_operator(bm_qlink1, theta*0.5);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE();
        inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE();

        fourier_transf_z2(bm_qlink0);
        DEBUG_CALL(printf("after fourier_transf_z2(bm_qlink0)\n"));
        DEBUG_READ_STATE();
        fourier_transf_z2(bm_qlink1);
        DEBUG_CALL(printf("after fourier_transf_z2(bm_qlink1)\n"));
        DEBUG_READ_STATE();
        fourier_transf_z2(bm_qlink2);
        DEBUG_CALL(printf("after fourier_transf_z2(bm_qlink2)\n"));
        DEBUG_READ_STATE();
        // fourier_transf_z2(bm_qlink3);
        // DEBUG_CALL(printf("after fourier_transf_z2(bm_qlink3)\n"));
        // DEBUG_READ_STATE();

        momentum_phase(bm_qlink0, 2*theta1, 2*theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink0, theta1, theta2)\n"));
        DEBUG_READ_STATE();
        momentum_phase(bm_qlink1, theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink1, theta1, theta2)\n"));
        DEBUG_READ_STATE();
        momentum_phase(bm_qlink2, theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink2, theta1, theta2)\n"));
        DEBUG_READ_STATE();
        // momentum_phase(bm_qlink3, theta1, theta2);
        // DEBUG_CALL(printf("after momentum_phase(bm_qlink3, theta1, theta2)\n"));
        // DEBUG_READ_STATE();
	

        // inverse_fourier_transf_z2(bm_qlink3);
        // DEBUG_CALL(printf("after inverse_fourier_transf_z2(bm_qlink3)\n"));
        // DEBUG_READ_STATE();
        inverse_fourier_transf_z2(bm_qlink2);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(bm_qlink2)\n"));
        DEBUG_READ_STATE();
        inverse_fourier_transf_z2(bm_qlink1);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(bm_qlink1)\n"));
        DEBUG_READ_STATE();
        inverse_fourier_transf_z2(bm_qlink0);
        DEBUG_CALL(printf("after inverse_fourier_transf_z2(bm_qlink0)\n"));
        DEBUG_READ_STATE();

        self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE();
        self_trace_operator(bm_qlink1, theta*0.5);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE();
        inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE();
//
//      TODO: merge V^{1/2} between different iterations
//        if(ti==(uint)(n-1)){
//            self_plaquette1();
//            DEBUG_CALL(printf("after self_plaquette1()\n"));
//            DEBUG_READ_STATE();
//            self_trace_operator(bm_qlink1, theta*2*0.5); // factor 2 because plaquette2 has the same value of plaquette 1 ; 0.5 because of Trotter
//            DEBUG_CALL(printf("after self_trace_operator()\n"));
//            DEBUG_READ_STATE();
//            inverse_self_plaquette1();
//            DEBUG_CALL(printf("after inverse_self_plaquette1()\n"));
//            DEBUG_READ_STATE();
//        }
    }
}



// qsa specifics
void qsa_init_state(){
    //TODO: implement
    throw std::runtime_error("ERROR: qsa_init_state() unimplemented!\n");
//    suqa::init_state();
//    suqa::apply_h(bm_spin[0]);
//    suqa::apply_h(bm_spin[1]);
//    suqa::apply_h(bm_spin[2]);
//    suqa::apply_cx(bm_spin[0], bm_spin_tilde[0]);
//    suqa::apply_cx(bm_spin[1], bm_spin_tilde[1]);
//    suqa::apply_cx(bm_spin[2], bm_spin_tilde[2]);
}

void evolution_szegedy(const double& t, const int& n){
    (void)t,(void)n;
    //TODO: implement
    throw std::runtime_error("ERROR: evolution_szegedy() unimplemented!\n");
//    (void)n;
//      DEBUG_CALL(std::cout<<"before evolution_szegedy()"<<std::endl);
//      DEBUG_READ_STATE();
//      DEBUG_CALL(std::cout<<"apply evolution_szegedy()"<<std::endl);
//    for (uint i = 0; i < 3; i++) {
//      suqa::apply_pauli_TP_rotation({bm_spin_tilde[(0+i)%3],bm_spin_tilde[(1+i)%3]}, {PAULI_X,PAULI_X}, -t);
//      DEBUG_CALL(std::cout<<"apply pauli 1 it "<<i<<std::endl);
//      DEBUG_READ_STATE();
//      suqa::apply_pauli_TP_rotation({bm_spin[(0+i)%3],bm_spin[(1+i)%3]}, {PAULI_X,PAULI_X}, t);
//      DEBUG_CALL(std::cout<<"apply pauli 2 it "<<i<<std::endl);
//      DEBUG_READ_STATE();
//    }
}

void evolution_measure(const double& t, const int& n){
    (void)t,(void)n;
    //TODO: implement
    throw std::runtime_error("ERROR: evolution_measure() unimplemented!\n");
}

void evolution_tracing(const double& t, const int& n){
    (void)t,(void)n;
    //TODO: implement
    throw std::runtime_error("ERROR: evolution_tracing() unimplemented!\n");
//    (void)n;
//  for (uint i = 0; i < 3; i++) {
//    suqa::apply_pauli_TP_rotation({bm_spin_tilde[(0+i)%3],bm_spin_tilde[(1+i)%3]}, {PAULI_X,PAULI_X}, -t);
//  }
//
}

/* Measure facilities */
const uint op_bits = 1; // 2^op_bits is the number of eigenvalues for the observable
const bmReg bm_op = bm_qlink1; // where the measure has to be taken
const std::vector<double> op_vals = {2.0,0.0}; // eigvals

 
// change basis to the observable basis somewhere in the system registers
void apply_measure_rotation(){
    self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
}

// inverse of the above function
void apply_measure_antirotation(){
    inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
}
// map the classical measure recorded in creg_vals
// to the corresponding value of the observable;
// there is no need to change it
double get_meas_opvals(const uint& creg_vals){
    return op_vals[creg_vals];
}

// actually perform the measure
// there is no need to change it
double measure_X(pcg& rgen){
    std::vector<uint> classics(op_bits);
    
    apply_measure_rotation();

    std::vector<double> rdoubs(op_bits);
    for(auto& el : rdoubs){
        el = rgen.doub();
    }
    suqa::measure_qbits(bm_op, classics, rdoubs);

    apply_measure_antirotation();

    uint meas = 0U;
    for(uint i=0; i<op_bits; ++i){
        meas |= (classics[i] << i);
    }

    return get_meas_opvals(meas);
}

/* Moves facilities */
#define NMoves 11

//std::vector<double> C_weightsums(NMoves);
std::vector<double> C_weightsums = {1./11, 2./11, 3./11, 4./11, 5./11, 6./11, 7./11, 8./11, 9./11, 10./11, 1.};
//#define HNMoves (NMoves>>1)


void apply_C(const uint &Ci, double rot_angle){
(void)rot_angle;
  switch(Ci){
  case 0U:
    suqa::apply_z(bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_z(bm_qlink1[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 1U:
    suqa::apply_z(bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_z(bm_qlink2[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 2U:
    suqa::apply_z(bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_z(bm_qlink0[0])\n"));
   DEBUG_READ_STATE();
    // suqa::apply_z(bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_z(bm_qlink3[0])\n"));
    // DEBUG_READ_STATE();
    break;
  case 3U:
    suqa::apply_y(bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink0[0])\n"));
    DEBUG_READ_STATE();
   // suqa::apply_y(bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_y(bm_qlink3[0])\n"));
    // DEBUG_READ_STATE();
   break;
  case 4U:
    suqa::apply_y(bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink1[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 5U:
    suqa::apply_y(bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink2[0])\n"));
   DEBUG_READ_STATE();
    break;
  case 6U:
    suqa::apply_x(bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink0[0])\n"));
    DEBUG_READ_STATE();
    // suqa::apply_y(bm_qlink3[0]);
    // DEBUG_CALL(printf("after apply_y(bm_qlink3[0])\n"));
    // DEBUG_READ_STATE();
    break;
  case 7U:
    suqa::apply_x(bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink1[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 8U:
    suqa::apply_x(bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink2[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 9U:
    suqa::apply_h(bm_qlink0[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink2[0])\n"));
   DEBUG_READ_STATE();
    break;
  case 10U:
    suqa::apply_h(bm_qlink1[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink2[0])\n"));
    DEBUG_READ_STATE();
    break;
  case 11U:
    suqa::apply_h(bm_qlink2[0]);
    DEBUG_CALL(printf("after apply_y(bm_qlink2[0])\n"));
    DEBUG_READ_STATE();
    break;
    
  default:
    throw std::runtime_error("ERROR: wrong move selection");
  }
}
void apply_C_inverse(const uint &Ci,double rot_angle){
  apply_C(Ci,rot_angle);
}


void qsa_apply_C(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: qsa_apply_C() unimplemented!\n");
//  suqa::apply_h(bm_spin_tilde[Ci]);
// suqa::apply_h(,bm_spin_tilde[(Ci+1)%3]);


  // suqa::apply_h(,bm_spin_tilde);
}

void qsa_apply_C_inverse(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: qsa_apply_C() unimplemented!\n");
//  if(Ci>2) throw std::runtime_error("ERROR: wrong move selection");
//  //suqa::apply_h(,bm_spin_tilde);
//  //suqa::apply_h(,bm_spin_tilde[(Ci+1)%3]);
//  suqa::apply_h(bm_spin_tilde[Ci]);
}

std::vector<double> get_C_weightsums(){ return C_weightsums; }

//std::vector<double> get_C_weightsums(){ 
//    static bool init_done=false;
//    // first initialization
//    if(not init_done){
//        for(int i=1; i<=NMoves; ++i){
//            C_weightsums[i-1]=i/(double)NMoves;
//        }
//        init_done=true;
//    }
//    return C_weightsums; }
//
