#pragma once
#include "Rand.hpp"
#include "io.hpp"
#include "suqa.cuh"
#include "system.cuh"

// defined in src/system.cu
void evolution_measure(const double& t, const int& n);
void evolution_szegedy(const double& t, const int& n);
void qsa_apply_C(const uint &Ci);
double measure_X(pcg& rgen);
void qsa_apply_C_inverse(const uint &Ci);

std::vector<double> get_C_weigthsums();

namespace qsa{

uint syst_qbits;
//uint state_levels;
uint ene_qbits;
uint szegedy_qbits;
uint ene_levels;
uint nqubits;
uint Dim;
unsigned long long iseed = 0ULL;
double t_phase_estimation;
double t_PE_factor;
double t_PE_shift;
double t_phase_estimation_szegedy;
double t_PE_factor_szegedy;
double t_PE_shift_szegedy;
int n_phase_estimation;
double beta;
uint W_mask_E;

uint gc_mask_szegedy;

pcg rangen;

std::vector<double> E_measures;
std::vector<double> X_measures;


std::vector<double> rphase_m;
std::vector<double> rphase_m_szegedy;

void fill_rphase(const uint& nqubits){
    rphase_m.resize(nqubits);
    uint c=1;
    for(uint i=0; i<nqubits; ++i){
        rphase_m[i] = (2.0*M_PI/(double)c);
        c<<=1;
    }
}
void fill_rphase_szegedy(const uint& nqubits){
    rphase_m_szegedy.resize(nqubits);
    uint c=1;
    for(uint i=0; i<nqubits; ++i){
        rphase_m_szegedy[i] = (2.0*M_PI/(double)c);
        c<<=1;
    }
}

void activate_gc_mask_szegedy(const bmReg& q_controls){
    qsa::gc_mask_szegedy=0U;
    for(const auto& q : q_controls)
        qsa::gc_mask_szegedy |= 1U << q;
	suqa::gc_mask=qsa::gc_mask_szegedy;
}

void deactivate_gc_mask_szegedy(){
    qsa::gc_mask_szegedy=0U;
    suqa::gc_mask=0U;
}

//XXX: ??
void activate_gc_mask_into_szegedy(const bmReg& q_controls){
   //suqa::gc_mask=suqa::gc_mask_szegedy;
    for(const auto& q : q_controls)
        suqa::gc_mask |= 1U << q;
}

void deactivate_gc_mask_into_szegedy(){
    suqa::gc_mask=qsa::gc_mask_szegedy;
}

// bitmap
std::vector<uint> bm_states;
std::vector<uint> bm_enes;

std::vector<uint> bm_szegedy;

uint bm_acc;

void fill_bitmap(){
    bm_states.resize(syst_qbits);
    bm_enes.resize(ene_qbits);
    bm_szegedy.resize(szegedy_qbits);
    uint c=0;
    for(uint i=0; i< syst_qbits; ++i)  bm_states[i] = c++;
    for(uint i=0; i< ene_qbits; ++i)    bm_enes[i] = c++;
    for(uint i=0; i< szegedy_qbits; ++i)    bm_szegedy[i] = c++;
    bm_acc = c;
    //test: provare a rimettere la rotazione con apply_W, attivo la maschera qui:
    for(uint i=0; i<ene_qbits; ++i){
        W_mask_E |= (1U << bm_enes[i]);
    }
}



void cevolution_measure(const double& t, const int& n, const uint& q_control, const bmReg& qstate){
    if(qstate.size()!=syst_qbits)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

    suqa::activate_gc_mask({q_control});
    evolution_measure(t, n);
    suqa::deactivate_gc_mask({q_control});
}



void cevolution_szegedy_PE(const double& t, const int& n, const uint& q_control, const bmReg& qstate){
    if(qstate.size()!=syst_qbits)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");
    DEBUG_CALL(std::cout<<"cevolution_szegedy()"<<std::endl);

    qsa::activate_gc_mask_into_szegedy({q_control});
    DEBUG_CALL(std::cout<<"activate_gc_mask_into_szegedy()\nglobal mask: "<<suqa::gc_mask<<", szegedy_mask: "<<gc_mask_szegedy<<std::endl);

    evolution_szegedy(t, n);

    qsa::deactivate_gc_mask_into_szegedy();
    DEBUG_CALL(std::cout<<"deactivate_gc_mask_into_szegedy()\nglobal mask: "<<suqa::gc_mask<<", szegedy_mask: "<<gc_mask_szegedy<<std::endl);

}


uint creg_to_uint(const std::vector<uint>& c_reg){
    if(c_reg.size()<1)
        throw std::runtime_error("ERROR: size of register zero.");

    uint ret = c_reg[0];
    for(uint j=1U; j<c_reg.size(); ++j)
       ret += c_reg[j] << j;

    return ret;
}

void reset_non_syst_qbits(){

    std::vector<double> rgenerates(ene_qbits);
    std::vector<double> rgenerates2(szegedy_qbits);


    for(auto& el : rgenerates) el = rangen.doub();
    suqa::apply_reset(bm_enes, rgenerates);

    for(auto& el : rgenerates2) el = rangen.doub();
    suqa::apply_reset(bm_szegedy, rgenerates2);

    suqa::apply_reset(bm_acc, rangen.doub());

}

void qsa_crm(const uint& q_control, const uint& q_target, const int& m){
    double rphase = (m>0) ? rphase_m[m] : rphase_m[-m];
    if(m<=0) rphase*=-1;



    suqa::apply_cu1(q_control, q_target, rphase);
}

//TODO: put in suqa
void qsa_qft(const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=qsize-1; outer_i>=0; outer_i--){
            suqa::apply_h(qact[outer_i]);

        for(int inner_i=outer_i-1; inner_i>=0; inner_i--){
            qsa_crm(qact[inner_i], qact[outer_i], +1+(outer_i-inner_i));

        }
    }
}

//TODO: put in suqa
void qsa_qft_inverse(const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=0; outer_i<qsize; outer_i++){
        for(int inner_i=0; inner_i<outer_i; inner_i++){
            qsa_crm(qact[inner_i], qact[outer_i], -1-(outer_i-inner_i));

        }
        suqa::apply_h(qact[outer_i]);

    }
}

void qsa_crm_szegedy(const uint& q_control, const uint& q_target, const int& m){
    double rphase = (m>0) ? rphase_m_szegedy[m] : rphase_m_szegedy[-m];
    if(m<=0) rphase*=-1;

    suqa::apply_cu1(q_control, q_target, rphase);
}

//TODO: put in suqa
void qsa_qft_szegedy(const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=qsize-1; outer_i>=0; outer_i--){
            suqa::apply_h(qact[outer_i]);

        for(int inner_i=outer_i-1; inner_i>=0; inner_i--){
            qsa_crm_szegedy(qact[inner_i], qact[outer_i], +1+(outer_i-inner_i));

        }
    }
}

//TODO: put in suqa
void qsa_qft_inverse_szegedy(const std::vector<uint>& qact){
    DEBUG_CALL(std::cout<<"qsa_qft_inverse_szegedy()"<<std::endl);
    int qsize = qact.size();
    for(int outer_i=0; outer_i<qsize; outer_i++){
        for(int inner_i=0; inner_i<outer_i; inner_i++){
            qsa_crm_szegedy(qact[inner_i], qact[outer_i], -1-(outer_i-inner_i));
            DEBUG_CALL(std::cout<<"after qsa_crm_szegedy() in qsa_qft_inverse_szegedy outer "<<outer_i<<", inner "<<inner_i<<std::endl);
            DEBUG_READ_STATE();

        }
        suqa::apply_h(qact[outer_i]);
        DEBUG_CALL(std::cout<<"after apply h in qsa_qft_inverse_szegedy"<<std::endl);
        DEBUG_READ_STATE();

    }
}
//in questa funzione registro sui bm_enes
void apply_phase_estimation(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){
  DEBUG_CALL(std::cout<<"apply_phase_estimation()"<<std::endl);


    suqa::apply_h(q_target);
  DEBUG_CALL(std::cout<<"after apply h in QPE"<<std::endl);
  DEBUG_READ_STATE();

    // apply CUs
    for(int trg = q_target.size() - 1; trg > -1; --trg){
        double powr = (double)(1U << (q_target.size()-1-trg));

        cevolution_szegedy_PE(-powr*t, powr*n, q_target[trg], q_state);
      DEBUG_CALL(std::cout<<"after cevolution_szegedy_PE() in QPE trg="<<trg<<std::endl);
      DEBUG_READ_STATE();

      suqa::apply_u1(q_target[trg], -powr*t*t_PE_shift);
      DEBUG_CALL(std::cout<<"after apply_u1() in QPE trg="<<trg<<std::endl);
      DEBUG_READ_STATE();
    }

    // apply QFT^{-1}
    qsa_qft_inverse(q_target);
    DEBUG_CALL(std::cout<<"after qsa_qft_inverse() in QPE"<<std::endl);
    DEBUG_READ_STATE();
}

void apply_phase_estimation_inverse(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){


    // apply QFT
    qsa_qft(q_target);


    // apply CUs
    for(uint trg = 0; trg < q_target.size(); ++trg){
        double powr = (double)(1U << (q_target.size()-1-trg));
        cevolution_szegedy_PE(powr*t, powr*n, q_target[trg], q_state);
        suqa::apply_u1(q_target[trg], powr*t*t_PE_shift);
    }

    suqa::apply_h(q_target);
}

//qui registro sui bm_szegedy per misurare
void apply_phase_estimation_measure(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){


    suqa::apply_h(q_target);


    // apply CUs
    for(int trg = q_target.size() - 1; trg > -1; --trg){
        double powr = (double)(1U << (q_target.size()-1-trg));

        cevolution_measure(-powr*t, powr*n, q_target[trg], q_state);

      suqa::apply_u1(q_target[trg], -powr*t*t_PE_shift_szegedy);
    }


    // apply QFT^{-1}
    qsa_qft_inverse_szegedy(q_target);

}

void apply_phase_estimation_measure_inverse(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){

    // apply QFT
    qsa_qft_szegedy(q_target);

    // apply CUs
    for(uint trg = 0; trg < q_target.size(); ++trg){
        double powr = (double)(1U << (q_target.size()-1-trg));
        suqa::apply_u1(q_target[trg], powr*t*t_PE_shift_szegedy);
        cevolution_measure(powr*t, powr*n, q_target[trg], q_state);
    }

    suqa::apply_h(q_target);
}

uint draw_C(){
    std::vector<double> C_weigthsums = get_C_weigthsums();
    double extract = rangen.doub();
    for(uint Ci =0U; Ci < C_weigthsums.size(); ++Ci){
        if(extract<C_weigthsums[Ci]){
            return Ci;
        }
    }
    return C_weigthsums.size();
}




std::vector<double> extract_rands(uint n){
    std::vector<double> ret(n);
    for(auto& el : ret){
        el = rangen.doub();
    }
    return ret;
}

//XXX: ??
void apply_ctrl_x(int x){
  int q=0,r=0;
  for(uint i=0;i<ene_qbits;i++){
    r=x%2;
    q=(x-r)/2;
    if(r==0) suqa::apply_x(bm_enes[i]);
    x=q;
  }
}
double eigenvalue_trans(int x_2d){
  double f_x_2d=0;
  f_x_2d= t_PE_shift+x_2d/(double)(t_PE_factor*ene_levels);
  return f_x_2d;
}
/*
void universal_rotation(ComplexVec& state,const double& beta_j){
  int x_2d=0,tmp1=0,tmp2=0;
  for(x_2d=0;x_2d< ene_levels;x_2d++){
    double tmp=eigenvalue_trans(x_2d);
    tmp1=x_2d;
    tmp2=x_2d;
    if(tmp>0){
      apply_ctrl_x(state,tmp1);
      suqa::activate_gc_mask_into_szegedy(bm_enes);
      suqa::apply_pauli_TP_rotation( state, {bm_acc}, {PAULI_Y},-acos(exp(-0.5*beta_j*tmp))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
      suqa::deactivate_gc_mask_into_szegedy();
      apply_ctrl_x(state,tmp2);
    }
  }
}
void universal_rotation_inverse(ComplexVec& state,const double& beta_j){
  int x_2d=0,tmp1=0,tmp2=0;
  for(x_2d=0;x_2d< ene_levels;x_2d++){
    double tmp=eigenvalue_trans(x_2d);
    tmp1=x_2d;
    tmp2=x_2d;
    if(tmp>0){
      apply_ctrl_x(state,tmp1);
      suqa::activate_gc_mask_into_szegedy(bm_enes);
      suqa::apply_pauli_TP_rotation( state, {bm_acc}, {PAULI_Y},acos(exp(-0.5*beta_j*tmp))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
      suqa::deactivate_gc_mask_into_szegedy();
      apply_ctrl_x(state,tmp2);
    }
  }
}
void universal_reflection(ComplexVec& state){
  int x_2d=0,tmp1=0,tmp2=0;
  for(x_2d=0;x_2d< ene_levels;x_2d++){
    double tmp=eigenvalue_trans(x_2d);
    tmp1=x_2d;
    tmp2=x_2d;
    if(tmp==0){
      apply_ctrl_x(state,tmp1);
      suqa::activate_gc_mask_into_szegedy(bm_enes);
      suqa::apply_x(state,   bm_acc);
      suqa::apply_u1(state,   bm_acc, M_PI);
      suqa::apply_x(state,   bm_acc);
      suqa::deactivate_gc_mask_into_szegedy();
      apply_ctrl_x(state,tmp2);
      }
    }
}
*/

void universal_rotation(const double& beta_j){
  int tmp1=0,tmp2=0;
  double tmp;
//  for(x_2d=0;x_2d< ene_levels;x_2d++){
 for(int x_2d=2;x_2d>1 ;x_2d--){
    tmp=eigenvalue_trans(x_2d);
   tmp1=x_2d;
    tmp2=x_2d;
  //  if(tmp>0){
      apply_ctrl_x(tmp1);
      qsa::activate_gc_mask_into_szegedy(bm_enes);
      suqa::apply_pauli_TP_rotation({bm_acc}, {PAULI_Y},-acos(exp(-0.5*beta_j*tmp))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
      qsa::deactivate_gc_mask_into_szegedy();
     apply_ctrl_x(tmp2);
    //}
  }
}
void universal_rotation_inverse(const double& beta_j){
  int tmp1=0,tmp2=0;
  //for(x_2d=0;x_2d< ene_levels;x_2d++){
      double tmp;
 for(int x_2d=2;x_2d>1 ;x_2d--){

        tmp=eigenvalue_trans(x_2d);
    tmp1=x_2d;
    tmp2=x_2d;
  //  if(tmp>0){
     apply_ctrl_x(tmp1);
      qsa::activate_gc_mask_into_szegedy(bm_enes);
      suqa::apply_pauli_TP_rotation({bm_acc}, {PAULI_Y},acos(exp(-0.5*beta_j*tmp))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
      qsa::deactivate_gc_mask_into_szegedy();
     apply_ctrl_x(tmp2);
    //}
  }
}
void universal_reflection(){
//  int x_2d=0,tmp1=0,tmp2=0;
  //for(x_2d=0;x_2d< ene_levels;x_2d++){

 //for(int x_2d=2;x_2d>1 ;x_2d--){
//  double  tmp=eigenvalue_trans(1);
  int  tmp1=1;
  int tmp2=1;
  //  if(tmp==0){
    apply_ctrl_x(tmp1);
    qsa::activate_gc_mask_into_szegedy(bm_enes);
    suqa::apply_x(bm_acc);
    suqa::apply_u1(bm_acc, M_PI);
    suqa::apply_x(bm_acc);
    qsa::deactivate_gc_mask_into_szegedy();
    apply_ctrl_x(tmp2);
      //}
  //  }
}
#ifdef GPU
//XXX: TESTTTTTTTTTTTTTTTTTTTTTT
__global__
void kernel_qsa_apply_W(double *const state_comp, uint len, uint q_acc,  uint dev_W_mask, uint dev_bm_enes, double c,double E_m, double PE_factor, uint levels){
    //XXX: since q_acc is the most relevant qubit, we split the cycle beforehand
    int i = blockDim.x*blockIdx.x + threadIdx.x+len/2;
    double fs1, fs2;
    while(i<len){
        // extract dE reading Eold and Enew
        uint j = i & ~(1U << q_acc);
        uint deltaE = (i & dev_W_mask) >> dev_bm_enes;
    //    printf("deltaE= %d\n", deltaE);
        double DE=		E_m+deltaE/(double)(PE_factor*levels);
        if(DE<0){
            fs1 = exp(-((DE)*c)/2.0);
            fs2 = sqrt(1.0 - fs1*fs1);
        }else{
            fs1 = 1.0;
            fs2 = 0.0;
        }
        double tmpval = state_comp[j];
        state_comp[j] = fs2*state_comp[j] + fs1*state_comp[i];
        state_comp[i] = fs1*tmpval        - fs2*state_comp[i]; // recall: i has 1 in the q_acc qbit
        i+=gridDim.x*blockDim.x;
    }
}

#endif

void apply_W(double const delta_beta){
#ifdef GPU
    qsa::kernel_qsa_apply_W<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(suqa::state.data_re, suqa::state.size(), bm_acc,  W_mask_E, bm_enes[0], delta_beta,t_PE_shift, t_PE_factor,ene_levels);
    qsa::kernel_qsa_apply_W<<<suqa::blocks,suqa::threads, 0, suqa::stream2>>>(suqa::state.data_im, suqa::state.size(), bm_acc,  W_mask_E, bm_enes[0], delta_beta,t_PE_shift, t_PE_factor,ene_levels);
    cudaDeviceSynchronize();
#else
    throw std::runtime_error("ERROR: qsa still not implemented on CPU only!!\n");
#endif
}

void apply_W_inverse(double const delta_beta){
    apply_W(delta_beta);
}
//TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT


void rotation(const double& beta_j){
//se bm_enes= 00 autovalore-4 se bm_enes[0]=0 e bm_enes[1]=1 autovalore 4
suqa::apply_x(bm_enes[0]);
	qsa::activate_gc_mask_into_szegedy(bm_enes);
suqa::apply_pauli_TP_rotation({bm_acc}, {PAULI_Y},-acos(exp(-2*beta_j))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
	qsa::deactivate_gc_mask_into_szegedy();
suqa::apply_x(bm_enes[0]);

}

void rotation_inverse(const double& beta_j){

suqa::apply_x(bm_enes[0]);
	qsa::activate_gc_mask_into_szegedy(bm_enes);
suqa::apply_pauli_TP_rotation({bm_acc}, {PAULI_Y},acos(exp(-2*beta_j))); //r_y(theta)=cos(theta/2)  2*acos(exp(-2*d_beta))
	qsa::deactivate_gc_mask_into_szegedy();
suqa::apply_x(bm_enes[0]);

}
void apply_ux(const double& beta_j,const uint &Ci){

	qsa_apply_C(Ci);

apply_phase_estimation(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);

	//rotation(beta_j);
 //apply_W(beta_j);
 universal_rotation(beta_j);
    DEBUG_CALL(std::cout<<"after universal_rotation() in apply_ux()"<<std::endl);
    DEBUG_READ_STATE();

apply_phase_estimation_inverse(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);

}
void apply_ux_inverse( const double& beta_j,const uint &Ci){

  apply_phase_estimation(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);

	//rotation_inverse(state,beta_j);
    //apply_W(beta_j);
     universal_rotation_inverse(beta_j);

  apply_phase_estimation_inverse(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);
  qsa_apply_C_inverse( Ci);
}
void apply_uy(const double& beta_j,const uint &Ci){
	apply_ux(beta_j,Ci);
	suqa::apply_x(bm_acc);
	qsa::activate_gc_mask_into_szegedy({bm_acc});
	for(uint j=0; j<bm_spin.size(); ++j) suqa::apply_swap(bm_spin[j],bm_spin_tilde[j]);
	qsa::deactivate_gc_mask_into_szegedy();
	suqa::apply_x(bm_acc);
}
void apply_uy_inverse(const double& beta_j,const uint &Ci){

	suqa::apply_x(bm_acc);
	qsa::activate_gc_mask_into_szegedy({bm_acc});
	for(uint j=0; j<bm_spin.size(); ++j) suqa::apply_swap(bm_spin[j],bm_spin_tilde[j]);
	qsa::deactivate_gc_mask_into_szegedy();
	suqa::apply_x(bm_acc);

  apply_ux_inverse(beta_j,Ci);
}
void reflection(){

	suqa::apply_mcx(bm_enes, {1U, 0U}, bm_acc);
	suqa::apply_mcu1(bm_enes, {1U, 0U}, bm_acc, M_PI);
	suqa::apply_mcx(bm_enes, {1U, 0U}, bm_acc);

}
void apply_lambda1(){

    apply_phase_estimation(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);
    universal_reflection();
    //	reflection(state);
    apply_phase_estimation_inverse(bm_states, bm_enes,  t_phase_estimation,  n_phase_estimation);
}

void apply_szegedy(const double& beta_j,const uint &Ci){
  apply_lambda1();
  apply_ux(beta_j,Ci);
  apply_uy_inverse(beta_j,Ci);
  apply_lambda1();
  apply_uy(beta_j, Ci);
  apply_ux_inverse(beta_j,Ci);
}

void cevolution_szegedy(const uint& q_control, const bmReg& qstate,const int& powr,const double& beta_j,const uint &Ci){
	if(qstate.size()!=qsa::syst_qbits)
		throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

	qsa::activate_gc_mask_szegedy({q_control});
		for(int mm=0; mm<powr; ++mm) {
			apply_szegedy(beta_j,Ci);
		}
	qsa::deactivate_gc_mask_szegedy();

}

void apply_PE_szegedy(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& beta_j,const uint &Ci){

  DEBUG_CALL(std::cout<<"apply_PE_szegedy()"<<std::endl);
	suqa::apply_h(q_target);
  DEBUG_CALL(std::cout<<"after apply h "<<std::endl);
  DEBUG_READ_STATE();


	// apply CUs
	for(int trg = q_target.size() - 1; trg > -1; --trg){

		double powr = (double)(1U << (q_target.size()-1-trg));
		cevolution_szegedy(q_target[trg], q_state,powr,beta_j,Ci);
          DEBUG_CALL(std::cout<<"after cevolution_szegedy()"<<std::endl);
          DEBUG_READ_STATE();


	}

	// apply QFT^{-1}
	qsa_qft_inverse_szegedy(q_target);
	//suqa::apply_h(state,q_target);
  DEBUG_CALL(std::cout<<"after qsa_qft_inverse_szegedy()"<<std::endl);
  DEBUG_READ_STATE();
}





void setup(){
    qsa::fill_rphase(qsa::ene_qbits+1);
    qsa::fill_rphase_szegedy(qsa::szegedy_qbits+1);
    qsa::fill_bitmap();
}

void clear(){
}


} // end of namespace qsa
