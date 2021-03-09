#pragma once
#include "Rand.hpp"
#include "io.hpp"
#include "suqa.cuh"
#include "complex_defines.cuh"


// defined in src/system.cu
void evolution(const double& t, const int& n);

double measure_X(pcg& rgen);

void apply_C(const uint &Ci);

void apply_C_inverse(const uint &Ci);

std::vector<double> get_C_weigthsums();
// end defs

namespace qms{

uint syst_qbits;
uint syst_levels;
uint ene_qbits;
uint ene_levels;
uint nqubits;
uint Dim;
uint max_reverse_attempts;
uint metro_steps;
uint reset_each;
unsigned long long iseed = 0ULL;
double t_phase_estimation;
double t_PE_factor;
double t_PE_shift;
int n_phase_estimation;
uint gCi;
uint c_acc = 0;

bool record_reverse=false;
std::vector<uint> reverse_counters;

pcg rangen;

////vector<double> energy_measures;
std::vector<double> X_measures;
std::vector<double> E_measures;

std::vector<double> rphase_m;
double c_factor;

void fill_rphase(const uint& nqubits){
    rphase_m.resize(nqubits);
    uint c=1;
    for(uint i=0; i<nqubits; ++i){
        rphase_m[i] = (2.0*M_PI/(double)c);
        c<<=1;
    }
} 

// bitmap
std::vector<uint> bm_syst;
std::vector<uint> bm_enes_old;
std::vector<uint> bm_enes_new;
uint bm_acc;


void fill_bitmap(){
    bm_syst.resize(syst_qbits);
    bm_enes_old.resize(ene_qbits);
    bm_enes_new.resize(ene_qbits);
    uint c=0;
    for(uint i=0; i< syst_qbits; ++i)  bm_syst[i] = c++;
    for(uint i=0; i< ene_qbits; ++i)    bm_enes_old[i] = c++;
    for(uint i=0; i< ene_qbits; ++i)    bm_enes_new[i] = c++;
    bm_acc = c;
}

// these are masks and precomputed values for apply_W
// on device they can be allocated in constant memory to speed accesses, but only if the qubits are few
uint W_mask;
uint W_mask_Eold;
uint W_mask_Enew;
//double *W_fs1, *W_fs2; // holds fs1 = exp(-b dE/2) and fs2 = sqrt(1-fs1^2)

void fill_W_utils(double beta, double t_PE_factor){
    c_factor = beta/(t_PE_factor*ene_levels);
    W_mask=0U;
    W_mask = (1U << bm_acc);
    W_mask_Eold = 0U;
    for(uint i=0; i<ene_qbits; ++i){
        W_mask |= (1U << bm_enes_old[i]) | (1U << bm_enes_new[i]);
        W_mask_Eold |= (1U << bm_enes_old[i]);
        W_mask_Enew |= (1U << bm_enes_new[i]);
    }
}


void cevolution(const double& t, const int& n, const uint& q_control, const bmReg& qsyst){
    if(qsyst.size()!=syst_qbits)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

    suqa::activate_gc_mask({q_control});
    
    evolution(t, n);

    suqa::deactivate_gc_mask({q_control});
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
    DEBUG_CALL(std::cout<<"\n\nBefore reset"<<std::endl);
    DEBUG_READ_STATE();
    std::vector<double> rgenerates(ene_qbits);

    for(auto& el : rgenerates) el = rangen.doub();
    suqa::apply_reset(bm_enes_old, rgenerates);

    DEBUG_CALL(std::cout<<"\n\nafter enes_old reset"<<std::endl);
    DEBUG_READ_STATE();

    for(auto& el : rgenerates) el = rangen.doub();
    suqa::apply_reset(bm_enes_new, rgenerates);

    DEBUG_CALL(std::cout<<"\n\nafter enes_new reset"<<std::endl);
    DEBUG_READ_STATE();

    suqa::apply_reset(bm_acc, rangen.doub());
    DEBUG_CALL(std::cout<<"\n\nAfter reset"<<std::endl);
    DEBUG_READ_STATE();
}

void qms_crm(const uint& q_control, const uint& q_target, const int& m){
    double rphase = (m>0) ? rphase_m[m] : rphase_m[-m];
    if(m<=0) rphase*=-1;

    DEBUG_CALL(std::cout<<"crm phase: m="<<m<<", rphase="<<rphase/M_PI<<" pi"<<std::endl);

    suqa::apply_cu1(q_control, q_target, rphase);
}

//TODO: put in suqa
void qms_qft(const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=qsize-1; outer_i>=0; outer_i--){
            suqa::apply_h(qact[outer_i]);
            DEBUG_CALL(std::cout<<"In qms_qft() after apply_h: outer_i = "<<outer_i<<std::endl);
            DEBUG_READ_STATE();
        for(int inner_i=outer_i-1; inner_i>=0; inner_i--){
            qms_crm(qact[inner_i], qact[outer_i], +1+(outer_i-inner_i));
            DEBUG_CALL(std::cout<<"In qms_qft() after crm: outer_i = "<<outer_i<<", inner_i = "<<inner_i<<std::endl);
            DEBUG_READ_STATE();
        }
    }
}

//TODO: put in suqa
void qms_qft_inverse(const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=0; outer_i<qsize; outer_i++){
        for(int inner_i=0; inner_i<outer_i; inner_i++){
            qms_crm(qact[inner_i], qact[outer_i], -1-(outer_i-inner_i));
            DEBUG_CALL(std::cout<<"In qms_qft_inverse() after crm: outer_i = "<<outer_i<<", inner_i = "<<inner_i<<std::endl);
            DEBUG_READ_STATE();
        }
        suqa::apply_h(qact[outer_i]);
        DEBUG_CALL(std::cout<<"In qms_qft_inverse() after apply_h: outer_i = "<<outer_i<<std::endl);
        DEBUG_READ_STATE();
    }
}

//TODO: put in suqa
void apply_phase_estimation(const std::vector<uint>& q_syst, const std::vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(std::cout<<"apply_phase_estimation()"<<std::endl);
    suqa::apply_h(q_target);
    DEBUG_CALL(std::cout<<"after qi_h(q_target)"<<std::endl);
    DEBUG_READ_STATE();

    // apply CUs
    for(int trg = q_target.size() - 1; trg > -1; --trg){
        double powr = (double)(1U << (q_target.size()-1-trg));
        DEBUG_CALL(std::cout<<"\nafter powr="<<powr<<", "<<"t="<<t<<", powr*t="<<powr*t<<std::endl);
        cevolution(-powr*t, powr*n, q_target[trg], q_syst);
        DEBUG_CALL(std::cout<<"\nafter trg="<<trg<<" cevolution"<<std::endl);
        DEBUG_READ_STATE();
        suqa::apply_u1(q_target[trg], -powr*t*t_PE_shift);
        DEBUG_CALL(std::cout<<"\nafter energy shift by Emin="<<t_PE_shift<<std::endl);
        DEBUG_READ_STATE();
    }
    DEBUG_CALL(std::cout<<"\nafter evolutions"<<std::endl);
    DEBUG_READ_STATE();
    
    // apply QFT^{-1}
    qms_qft_inverse(q_target); 

}

//TODO: put in suqa
void apply_phase_estimation_inverse(const std::vector<uint>& q_syst, const std::vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(std::cout<<"apply_phase_estimation_inverse()"<<std::endl);

    // apply QFT
    qms_qft(q_target); 
    DEBUG_CALL(std::cout<<"\nafter qft"<<std::endl);
    DEBUG_READ_STATE();

    // apply CUs
    for(uint trg = 0; trg < q_target.size(); ++trg){
        double powr = (double)(1U << (q_target.size()-1-trg));
        cevolution(powr*t, powr*n, q_target[trg], q_syst);
        suqa::apply_u1(q_target[trg], powr*t*t_PE_shift);
    }

    DEBUG_CALL(std::cout<<"\nafter evolutions"<<std::endl);
    DEBUG_READ_STATE();
    
    suqa::apply_h(q_target);
}


void apply_Phi_old(){ apply_phase_estimation(bm_syst, bm_enes_old, t_phase_estimation, n_phase_estimation); }
void apply_Phi_old_inverse(){ apply_phase_estimation_inverse(bm_syst, bm_enes_old, t_phase_estimation, n_phase_estimation); }
void apply_Phi(){ apply_phase_estimation(bm_syst, bm_enes_new, t_phase_estimation, n_phase_estimation); }
void apply_Phi_inverse(){ apply_phase_estimation_inverse(bm_syst, bm_enes_new, t_phase_estimation, n_phase_estimation); }


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


//TODO: put generic oracle builder in suqa
#ifdef GPU
__global__
void kernel_qms_apply_W(double *const state_comp, uint len, uint q_acc, uint dev_W_mask_Eold, uint dev_bm_enes_old, uint dev_W_mask_Enew, uint dev_bm_enes_new, double c){
    //XXX: since q_acc is the most significative qubit, we split the cycle beforehand
    int i = blockDim.x*blockIdx.x + threadIdx.x+len/2;    
    double fs1, fs2;
    while(i<len){
        // extract dE reading Eold and Enew
        uint j = i & ~(1U << q_acc);
        uint Eold = (i & dev_W_mask_Eold) >> dev_bm_enes_old;
        uint Enew = (i & dev_W_mask_Enew) >> dev_bm_enes_new;
        if(Enew>Eold){
            fs1 = exp(-((Enew-Eold)*c)/2.0);
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
#else // CPU
void func_qms_apply_W(uint q_acc, uint dev_W_mask_Eold, uint dev_bm_enes_old, uint dev_W_mask_Enew, uint dev_bm_enes_new, double c){
    double fs1, fs2;
#ifdef SPARSE
    std::vector<uint> new_actives; // instead of removing from suqa::actives, replace with new actives
    std::vector<uint> visited;
	for (auto& i : suqa::actives){
        if((i & suqa::gc_mask) == suqa::gc_mask){ 
            // extract dE reading Eold and Enew
            uint i_0 = i & ~(1U << q_acc);
            uint i_1 = i_0 | (1U << q_acc);
            if (std::find(visited.begin(), visited.end(), i_0) == visited.end()) { // apply only once
                //extract energies from other registers
                uint Eold = (i_0 & dev_W_mask_Eold) >> dev_bm_enes_old;
                uint Enew = (i_0 & dev_W_mask_Enew) >> dev_bm_enes_new;
                if(Enew>Eold){
                    fs1 = exp(-((Enew-Eold)*c)/2.0);
                    fs2 = sqrt(1.0 - fs1*fs1);
                }else{
                    fs1 = 1.0;
                    fs2 = 0.0;
                }
                double tmpval = suqa::state.data_re[i_0];
                suqa::state.data_re[i_0] = fs2*suqa::state.data_re[i_0] + fs1*suqa::state.data_re[i_1];
                suqa::state.data_re[i_1] = fs1*tmpval        - fs2*suqa::state.data_re[i_1];
                tmpval = suqa::state.data_im[i_0];
                suqa::state.data_im[i_0] = fs2*suqa::state.data_im[i_0] + fs1*suqa::state.data_im[i_1];
                suqa::state.data_im[i_1] = fs1*tmpval        - fs2*suqa::state.data_im[i_1];

                if(norm(suqa::state.data_re[i_0],suqa::state.data_im[i_0])>1e-8)
                    new_actives.push_back(i_0);

                if(norm(suqa::state.data_re[i_1],suqa::state.data_im[i_1])>1e-8)
                    new_actives.push_back(i_1);
               
                visited.push_back(i_0); 
            }
        }else{
			new_actives.push_back(i);
        }
    }
#else
    //XXX: since q_acc is the most significative qubit, we split the cycle beforehand
    for (uint i = suqa::state.size()/2; i < suqa::state.size(); ++i) {
        // extract dE reading Eold and Enew
        uint j = i & ~(1U << q_acc);
        uint Eold = (i & dev_W_mask_Eold) >> dev_bm_enes_old;
        uint Enew = (i & dev_W_mask_Enew) >> dev_bm_enes_new;
        if(Enew>Eold){
            fs1 = exp(-((Enew-Eold)*c)/2.0);
            fs2 = sqrt(1.0 - fs1*fs1);
        }else{
            fs1 = 1.0;
            fs2 = 0.0;
        }
        double tmpval = suqa::state.data_re[j];
        suqa::state.data_re[j] = fs2*suqa::state.data_re[j] + fs1*suqa::state.data_re[i];
        suqa::state.data_re[i] = fs1*tmpval        - fs2*suqa::state.data_re[i]; // recall: i has 1 in the q_acc qbit 
        tmpval = suqa::state.data_im[j];
        suqa::state.data_im[j] = fs2*suqa::state.data_im[j] + fs1*suqa::state.data_im[i];
        suqa::state.data_im[i] = fs1*tmpval        - fs2*suqa::state.data_im[i]; // recall: i has 1 in the q_acc qbit 
    }
#endif
}
#endif

void apply_W(){
    DEBUG_CALL(std::cout<<"\n\nApply W"<<std::endl);

#ifdef GPU
    qms::kernel_qms_apply_W<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(suqa::state.data_re, suqa::state.size(), bm_acc, W_mask_Eold, bm_enes_old[0], W_mask_Enew, bm_enes_new[0], c_factor);
    qms::kernel_qms_apply_W<<<suqa::blocks,suqa::threads, 0, suqa::stream2>>>(suqa::state.data_im, suqa::state.size(), bm_acc, W_mask_Eold, bm_enes_old[0], W_mask_Enew, bm_enes_new[0], c_factor);
    cudaDeviceSynchronize();
#else
    qms::func_qms_apply_W(bm_acc, W_mask_Eold, bm_enes_old[0], W_mask_Enew, bm_enes_new[0], c_factor);
#endif
}

void apply_W_inverse(){
    apply_W();
}

void apply_U(){
    DEBUG_CALL(std::cout<<"\n\nApply U"<<std::endl);
    apply_C(gCi);
    DEBUG_CALL(std::cout<<"\n\nAfter apply C = "<<gCi<<std::endl);
    DEBUG_READ_STATE();

    apply_Phi();
    DEBUG_CALL(std::cout<<"\n\nAfter second phase estimation"<<std::endl);
    DEBUG_READ_STATE();

    apply_W();
    DEBUG_CALL(std::cout<<"\n\nAfter apply W"<<std::endl);
    DEBUG_READ_STATE();
}

void apply_U_inverse(){
    apply_W_inverse();
    DEBUG_CALL(std::cout<<"\n\nAfter apply W inverse"<<std::endl);
    DEBUG_READ_STATE();
    apply_Phi_inverse();
    DEBUG_CALL(std::cout<<"\n\nAfter inverse second phase estimation"<<std::endl);
    DEBUG_READ_STATE();
    apply_C_inverse(gCi);
    DEBUG_CALL(std::cout<<"\n\nAfter apply C inverse = "<<gCi<<std::endl);
    DEBUG_READ_STATE();
}


std::vector<double> extract_rands(uint n){
    std::vector<double> ret(n);
    for(auto& el : ret){
        el = rangen.doub();
    }
    return ret;
}


int metro_step(bool take_measure){

    // return values:
    // 1 -> step accepted, not measured
    // 2 -> step accepted, measured
    // 3 -> step rejected and restored, not measured
    // 4 -> step rejected and restored, measured
    // -1 -> step rejected non restored 
    int ret=0;
    
    DEBUG_CALL(std::cout<<"initial state"<<std::endl);
    DEBUG_READ_STATE();
    reset_non_syst_qbits();
    DEBUG_CALL(std::cout<<"state after reset"<<std::endl);
    DEBUG_READ_STATE();
    apply_Phi_old();
    DEBUG_CALL(std::cout<<"\n\nAfter first phase estimation"<<std::endl);
    DEBUG_READ_STATE();
    std::vector<uint> c_E_news(ene_qbits,0), c_E_olds(ene_qbits,0);
    suqa::measure_qbits(bm_enes_old, c_E_olds, extract_rands(ene_qbits));
    DEBUG_CALL(std::cout<<"\n\nAfter measure on bm_enes_old"<<std::endl);
    DEBUG_READ_STATE();
    DEBUG_CALL(double tmp_E=t_PE_shift+creg_to_uint(c_E_olds)/(double)(t_PE_factor*ene_levels));
    DEBUG_CALL(std::cout<<"  energy measure: "<<tmp_E<<std::endl); 

    gCi = draw_C();
    DEBUG_CALL(std::cout<<"\n\ndrawn C = "<<gCi<<std::endl);
    apply_U();


    suqa::measure_qbit(bm_acc, c_acc, rangen.doub());

    if (c_acc == 1U){
        DEBUG_CALL(std::cout<<"accepted"<<std::endl);
        double Enew_meas_d;
        DEBUG_CALL(std::cout<<"Measuring energy new"<<std::endl);
        suqa::measure_qbits(bm_enes_new, c_E_news, extract_rands(ene_qbits));
        DEBUG_CALL(double tmp_E=t_PE_shift+creg_to_uint(c_E_news)/(double)(t_PE_factor*ene_levels));
        DEBUG_CALL(std::cout<<"  energy measure: "<<creg_to_uint(c_E_news)<<" --> "<<tmp_E<<"\nstate after measure:"<<std::endl); 
        DEBUG_CALL(std::cout<<"t_PE_factor = "<<t_PE_factor<<std::endl); 
        DEBUG_READ_STATE()
        apply_Phi_inverse();
        if(take_measure){
            Enew_meas_d = t_PE_shift+creg_to_uint(c_E_news)/(double)(t_PE_factor*ene_levels); // -1 + 3/(3/4)= -1 + 4
            E_measures.push_back(Enew_meas_d);
            for(uint ei=0U; ei<ene_qbits; ++ei){
                suqa::apply_reset(bm_enes_new[ei],rangen.doub());
            }
            X_measures.push_back(measure_X(rangen));
            DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
            DEBUG_CALL(std::cout<<"\n\nAfter X measure"<<std::endl);
            DEBUG_READ_STATE();
            DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
////           reset_non_syst_qbits();
            for(uint ei=0U; ei<ene_qbits; ++ei)
                suqa::apply_reset(bm_enes_new[ei],rangen.doub());
            apply_Phi();
            suqa::measure_qbits(bm_enes_new, c_E_news, extract_rands(ene_qbits));
            DEBUG_CALL(std::cout<<"\n\nAfter E recollapse"<<std::endl);
            DEBUG_READ_STATE();
            apply_Phi_inverse();

            ret = 2; // step accepted, measured
        }else{
            ret = 1; // step accepted, not measured
        }
        return ret;
    }
    //else

    DEBUG_CALL(std::cout<<"rejected; restoration cycle:"<<std::endl);
    apply_U_inverse();
    
    DEBUG_CALL(std::cout<<"\n\nBefore reverse attempts"<<std::endl);
    DEBUG_READ_STATE();
    uint iters = 0;
    while(iters < max_reverse_attempts){
        apply_Phi();
        uint Eold_meas, Enew_meas;
        double Eold_meas_d;
        std::vector<uint> c_E_olds(ene_qbits,0), c_E_news(ene_qbits,0);
        suqa::measure_qbits(bm_enes_old, c_E_olds, extract_rands(ene_qbits));
        Eold_meas = creg_to_uint(c_E_olds);
        Eold_meas_d = t_PE_shift+Eold_meas/(double)(t_PE_factor*ene_levels);
        suqa::measure_qbits(bm_enes_new, c_E_news, extract_rands(ene_qbits));
        Enew_meas = creg_to_uint(c_E_news);
        apply_Phi_inverse();
        
        if(Eold_meas == Enew_meas){
            DEBUG_CALL(std::cout<<"  accepted restoration ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<std::endl); 
            if(take_measure){
                E_measures.push_back(Eold_meas_d);
                DEBUG_CALL(std::cout<<"  energy measure : "<<Eold_meas_d<<std::endl); 
                DEBUG_CALL(std::cout<<"\n\nBefore X measure"<<std::endl);
                DEBUG_READ_STATE();
                for(uint ei=0U; ei<ene_qbits; ++ei)
                    suqa::apply_reset(bm_enes_new[ei],rangen.doub());

                X_measures.push_back(measure_X(rangen));
                DEBUG_CALL(std::cout<<"\n\nAfter X measure"<<std::endl);
                DEBUG_READ_STATE();
                DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
                for(uint ei=0U; ei<ene_qbits; ++ei)
                    suqa::apply_reset(bm_enes_new[ei],rangen.doub());

                apply_Phi();
                suqa::measure_qbits(bm_enes_new, c_E_news, extract_rands(ene_qbits));
                DEBUG_CALL(std::cout<<"\n\nAfter E recollapse"<<std::endl);
                DEBUG_READ_STATE();
                apply_Phi_inverse();

                ret=4;
            }else{
                ret=3;
            }
            break;
        }
        //else
        DEBUG_CALL(std::cout<<"  rejected ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<std::endl); 
        uint c_acc_trash;
        apply_U(); 
        suqa::measure_qbit(bm_acc, c_acc_trash, rangen.doub()); 
        apply_U_inverse(); 

        iters++;
    }

    if(record_reverse){
        reverse_counters.push_back(iters);
    }

    if (iters == max_reverse_attempts){
        DEBUG_CALL(std::cout<<("not converged in "+std::to_string(max_reverse_attempts)+" steps :(")<<std::endl);

        ret = -1;
        return ret;
    }else{
        return ret;
    }
}

void setup(double beta){
    qms::fill_rphase(qms::ene_qbits+1);
    qms::fill_bitmap();
    qms::fill_W_utils(beta, qms::t_PE_factor);

#if defined(GPU) & !defined(NDEBUG)
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_re,qms::Dim*sizeof(double),cudaHostAllocDefault));
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_im,qms::Dim*sizeof(double),cudaHostAllocDefault));
#endif
}

void clear(){
#if defined(GPU) & !defined(NDEBUG)
    HANDLE_CUDACALL(cudaFreeHost(host_state_re));
    HANDLE_CUDACALL(cudaFreeHost(host_state_im));
#endif
}


}
