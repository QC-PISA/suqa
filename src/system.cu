#include "system.cuh"

/* d4 gauge theory - two plaquettes
 
   link state 3 qubits
   system state: 4 links -> 12 qubits
   +1 ancillary qubit

 */

double g_beta;

__inline__ double f1(double b){
    return log((3+cosh(2.*b))/(2*sinh(b)*sinh(b)));
}

__inline__ double f2(double b){
    return -log(tanh(b));
}

void init_state(){
    suqa::init_state();

    suqa::apply_h(bm_qlink0[0]);
    suqa::apply_cx(bm_qlink0[0], bm_qlink3[0]);
    suqa::apply_h(bm_qlink0[1]);
    suqa::apply_cx(bm_qlink0[1], bm_qlink3[1]);
    suqa::apply_h(bm_qlink0[2]);
    suqa::apply_cx(bm_qlink0[2], bm_qlink3[2]);
    suqa::apply_mcx({bm_qlink3[0], bm_qlink3[2]}, {0U,1U}, bm_qlink3[1]);
}


/* Quantum evolutor of the state */

void inversion(const bmReg& q){
    suqa::apply_mcx({q[0],q[2]},{1U,0U},q[1]); 
}

void left_multiplication(const bmReg& qr1, const bmReg& qr2){
    suqa::apply_cx(qr1[1], qr2[1]);
    suqa::apply_mcx({qr1[0], qr2[0]}, qr2[1]);
    suqa::apply_cx(qr1[0], qr2[0]);
    suqa::apply_mcx({qr1[0], qr2[2]}, qr2[1]);
    suqa::apply_cx(qr1[2], qr2[2]);
}

void self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    inversion(qr1);
    left_multiplication(qr1, qr0);
    inversion(qr1);
    inversion(qr2);
    left_multiplication(qr2, qr0);
    inversion(qr2);
    left_multiplication(qr3, qr0);
}

void inverse_self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3){
    inversion(qr3);
    left_multiplication(qr3, qr0);
    inversion(qr3);
    left_multiplication(qr2, qr0);
    left_multiplication(qr1, qr0);
}

void cphases(uint qaux, uint q0b, double alpha1, double alpha2){
    suqa::apply_cx(qaux, q0b);
    suqa::apply_cu1(q0b, qaux, alpha1, 1U);
    suqa::apply_cx(qaux, q0b);
    suqa::apply_cu1(q0b, qaux, alpha2, 1U);
}

void self_trace_operator(const bmReg& qr, const uint& qaux, double th){
    suqa::apply_mcx({qr[0],qr[2]}, {0U,0U}, qaux); 
    cphases(qaux, qr[1], th, -th);
    suqa::apply_mcx({qr[0],qr[2]}, {0U,0U}, qaux); 
}

void fourier_transf_d4(const bmReg& qr){
    suqa::apply_cx(qr[2], qr[0]);
    suqa::apply_cx(qr[0], qr[2]);
    suqa::apply_tdg(qr[2]);
    suqa::apply_tdg(qr[2]);
    suqa::apply_cx(qr[1], qr[2]);
    suqa::apply_h(qr[0]);
    suqa::apply_h(qr[1]);
    suqa::apply_h(qr[2]);
    suqa::apply_t(qr[1]);
    suqa::apply_tdg(qr[2]);
    suqa::apply_cx(qr[1], qr[2]);
    suqa::apply_cx(qr[0], qr[1]);
    suqa::apply_h(qr[1]);
    suqa::apply_t(qr[1]);
    suqa::apply_t(qr[1]);
    suqa::apply_h(qr[1]);
}


void inverse_fourier_transf_d4(const bmReg& qr){
    suqa::apply_h(qr[1]);
    suqa::apply_tdg(qr[1]);
    suqa::apply_tdg(qr[1]);
    suqa::apply_h(qr[1]);
    suqa::apply_cx(qr[0], qr[1]);
    suqa::apply_cx(qr[1], qr[2]);
    suqa::apply_t(qr[2]);
    suqa::apply_tdg(qr[1]);
    suqa::apply_h(qr[0]);
    suqa::apply_h(qr[1]);
    suqa::apply_h(qr[2]);
    suqa::apply_cx(qr[1], qr[2]);
    suqa::apply_t(qr[2]);
    suqa::apply_t(qr[2]);
    suqa::apply_cx(qr[0], qr[2]);
    suqa::apply_cx(qr[2], qr[0]);
}

void momentum_phase(const bmReg& qr, const uint& qaux, double th1, double th2){
    suqa::apply_mcx(qr, {0U,0U,0U}, qaux);
    DEBUG_CALL(printf("\tafter suqa::apply_mcx(qr, {0U,0U,0U}, qaux)\n"));
    DEBUG_READ_STATE();
    suqa::apply_cx(qaux, qr[2]);
    suqa::apply_cu1(qaux, qr[2], th1);
    suqa::apply_cx(qaux, qr[2]);
    DEBUG_CALL(printf("\tafter suqa::apply_cu1(qaux, qr[2], th1, 0U)\n"));
    DEBUG_READ_STATE();
    suqa::apply_u1(qr[2], th2);
    DEBUG_CALL(printf("\tafter suqa::apply_u1(qr[2], th2)\n"));
    DEBUG_READ_STATE();
    suqa::apply_mcx(qr, {0U,0U,0U}, qaux);
    DEBUG_CALL(printf("\tafter suqa::apply_mcx(qr, {0U,0U,0U}, qaux)\n"));
    DEBUG_READ_STATE();
}

void evolution(const double& t, const int& n){
    const double dt = t/(double)n;

    const double theta1 = dt*f1(g_beta);
    const double theta2 = dt*f2(g_beta);
    const double theta = 2*dt*g_beta;
//    printf("g_beta = %.16lg, dt = %.16lg, thetas: %.16lg %.16lg %.16lg\n", g_beta, dt, theta1, theta2, theta);

    for(uint ti=0; ti<(uint)n; ++ti){
        self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE();
        self_trace_operator(bm_qlink1, bm_qaux[0], theta);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE();
        inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE();

        self_plaquette(bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        DEBUG_CALL(printf("after self_plaquette()\n"));
        DEBUG_READ_STATE();
        self_trace_operator(bm_qlink2, bm_qaux[0], theta);
        DEBUG_CALL(printf("after self_trace_operator()\n"));
        DEBUG_READ_STATE();
        inverse_self_plaquette(bm_qlink2, bm_qlink3, bm_qlink1, bm_qlink3);
        DEBUG_CALL(printf("after inverse_self_plaquette()\n"));
        DEBUG_READ_STATE();

        fourier_transf_d4(bm_qlink0);
        DEBUG_CALL(printf("after fourier_transf_d4(bm_qlink0)\n"));
        DEBUG_READ_STATE();
        fourier_transf_d4(bm_qlink1);
        DEBUG_CALL(printf("after fourier_transf_d4(bm_qlink1)\n"));
        DEBUG_READ_STATE();
        fourier_transf_d4(bm_qlink2);
        DEBUG_CALL(printf("after fourier_transf_d4(bm_qlink2)\n"));
        DEBUG_READ_STATE();
        fourier_transf_d4(bm_qlink3);
        DEBUG_CALL(printf("after fourier_transf_d4(bm_qlink3)\n"));
        DEBUG_READ_STATE();

        momentum_phase(bm_qlink0, bm_qaux[0], theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink0, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE();
        momentum_phase(bm_qlink1, bm_qaux[0], theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink1, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE();
        momentum_phase(bm_qlink2, bm_qaux[0], theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink2, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE();
        momentum_phase(bm_qlink3, bm_qaux[0], theta1, theta2);
        DEBUG_CALL(printf("after momentum_phase(bm_qlink3, bm_qaux[0], theta1, theta2)\n"));
        DEBUG_READ_STATE();


        inverse_fourier_transf_d4(bm_qlink3);
        DEBUG_CALL(printf("after inverse_fourier_transf_d4(bm_qlink3)\n"));
        DEBUG_READ_STATE();
        inverse_fourier_transf_d4(bm_qlink2);
        DEBUG_CALL(printf("after inverse_fourier_transf_d4(bm_qlink2)\n"));
        DEBUG_READ_STATE();
        inverse_fourier_transf_d4(bm_qlink1);
        DEBUG_CALL(printf("after inverse_fourier_transf_d4(bm_qlink1)\n"));
        DEBUG_READ_STATE();
        inverse_fourier_transf_d4(bm_qlink0);
        DEBUG_CALL(printf("after inverse_fourier_transf_d4(bm_qlink0)\n"));
        DEBUG_READ_STATE();
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
const uint op_bits = 3; // 2^op_bits is the number of eigenvalues for the observable
const bmReg bm_op = bm_qlink1; // where the measure has to be taken
const std::vector<double> op_vals = {2.0,0.0,-2.0, 0.0,0.0,0.0,0.0,0.0}; // eigvals

 
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

std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};

void apply_C(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: apply_C() unimplemented!\n");
}

void apply_C_inverse(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: apply_C_inverse() unimplemented!\n");
}

void qsa_apply_C(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: qsa_apply_C() unimplemented!\n");
//  suqa::apply_h(bm_spin_tilde[Ci]);
// suqa::apply_h(state,bm_spin_tilde[(Ci+1)%3]);


  // suqa::apply_h(state,bm_spin_tilde);
}

void qsa_apply_C_inverse(const uint &Ci){
    (void)Ci;
    //TODO: implement
    throw std::runtime_error("ERROR: qsa_apply_C() unimplemented!\n");
//  if(Ci>2) throw std::runtime_error("ERROR: wrong move selection");
//  //suqa::apply_h(state,bm_spin_tilde);
//  //suqa::apply_h(state,bm_spin_tilde[(Ci+1)%3]);
//  suqa::apply_h(bm_spin_tilde[Ci]);
}

std::vector<double> get_C_weigthsums(){ return C_weigthsums; }

