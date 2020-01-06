#pragma once
#include "Rand.hpp"
#include "io.hpp"
#include "suqa.cuh"
#include "complex_defines.cuh"

#if !defined(NDEBUG) 
#if defined(CUDA_HOST)
#define DEBUG_READ_STATE() {\
    sparse_print((double*)qms::gState.data, qms::gState.size()); \
}
#else
Complex *host_state;
#define DEBUG_READ_STATE() {\
    cudaMemcpy(host_state,gState.data,gState.size()*sizeof(Complex),cudaMemcpyDeviceToHost); \
    sparse_print((double*)host_state,gState.size()); \
} 
#endif
#else
#define DEBUG_READ_STATE()
#endif

// defined in src/evolution.cpp
void cevolution(ComplexVec& state, const double& t, const int& n, const uint& q_control, const std::vector<uint>& qstate);

void fill_meas_cache(const std::vector<uint>& bm_states, const std::string opstem);

void apply_measure_rotation(ComplexVec& state);
void apply_measure_antirotation(ComplexVec& state);
double get_meas_opvals(const uint& creg_vals);

void apply_C(ComplexVec& state, const std::vector<uint>& bm_states, const uint &Ci);

void apply_C_inverse(ComplexVec& state, const std::vector<uint>& bm_states, const uint &Ci);

std::vector<double> get_C_weigthsums();
// end defs

namespace qms{

uint state_qbits;
uint state_levels;
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
int n_phase_estimation;
uint gCi;
uint c_acc = 0;
std::string Xmatstem="";

bool record_reverse=false;
std::vector<uint> reverse_counters;

pcg rangen;

ComplexVec gState;
////vector<double> energy_measures;
std::vector<double> X_measures;
std::vector<double> E_measures;


std::vector<Complex> rphase_m;

void fill_rphase(const uint& nlevels){
    rphase_m.resize(nlevels);
    uint c=1;
    for(uint i=0; i<nlevels; ++i){
        rphase_m[i].x = cos((2.0*M_PI/(double)c));
        rphase_m[i].y = sin((2.0*M_PI/(double)c));
        c<<=1;
	//printf("rphase_m[i] %.12lf %.12lf\n", real(rphase_m[i]), imag(rphase_m[i]));
    }
} 


// bitmap
std::vector<uint> bm_states;
std::vector<uint> bm_enes_old;
std::vector<uint> bm_enes_new;
uint bm_acc;


void fill_bitmap(){
    bm_states.resize(state_qbits);
    bm_enes_old.resize(ene_qbits);
    bm_enes_new.resize(ene_qbits);
    uint c=0;
    for(uint i=0; i< state_qbits; ++i)  bm_states[i] = c++;
    for(uint i=0; i< ene_qbits; ++i)    bm_enes_old[i] = c++;
    for(uint i=0; i< ene_qbits; ++i)    bm_enes_new[i] = c++;
    bm_acc = c;
}

// these are masks and precomputed values for apply_W
// on device they can be allocated in constant memory to speed accesses, but only if the qubits are few
uint W_mask;
std::vector<double> W_fs1, W_fs2; // holds fs1 = exp(-b dE/2) and fs2 = sqrt(1-fs1^2)
std::vector<std::vector<uint>> W_case_masks;

// max 12 qubits means 4096 energy levels, means 32KB, let's allocate 64KB for the composite W_fs1 and W_fs2 variables
__constant__ double dev_W_fs1[4096];
__constant__ double dev_W_fs2[4096];

// here the counting proceeds as for an upper/lower triangular matrix
// e.g.:
// qbits | ene_levels | size
//     1 |          2 |    1         = 1
//     2 |          4 |    3+2+1     = 3*(3+1)/2 = 6
//     3 |          8 |    7+6+5...1 = 8*(8+1)/2 = 36
//     q |        2^q |   (2^q+1)*2^(q-1)
// let us assume max Q qubits for the energy -> 4B*(2^12+1)*2^11 = 32MB, TOO MUCH!!
// only using 7 or less qubits for the energy registers (~32KB) can be sustained, but 
// we would obtain marginal advantage, since things would become more cpu efficients then...
// so... no constant memory, let's use the global one!
// (XXX: maybe texture memory? it is 2^27B -> ~130MB)
uint * dev_W_case_masks; //[8390656];

__host__ __device__ static __inline__ uint uptri_sum_formula(uint i){ return (i*i-i)/2; }

void fill_W_utils(double beta, double t_PE_factor){
    W_mask=0U;
    W_mask = (1U << bm_acc);
    for(uint i=0; i<ene_qbits; ++i)
        W_mask |= (1U << bm_enes_old[i]) | (1U << bm_enes_new[i]);

    // energy goes from 0 to (ene_levels-1)*t_PE_factor
    W_fs1.resize(ene_levels);
    W_fs2.resize(ene_levels);
    double c = beta/(t_PE_factor*ene_levels);
    for(uint i=0; i<ene_levels; ++i){
        W_fs1[i] = exp(-(double)(i*c)/2);
        W_fs2[i] = sqrt(1.-W_fs1[i]*W_fs1[i]);
    }

    // mask cases
#if !defined(CUDA_HOST)
    uint W_case_masks_size = uptri_sum_formula(ene_levels);
    uint host_W_case_masks[W_case_masks_size];
#endif
    W_case_masks = std::vector<std::vector<uint>>(ene_levels); 
    for(uint i=1; i<ene_levels; ++i){ // all possible (non-trivial) values of Delta E
        W_case_masks[i] = std::vector<uint>(ene_levels-i,(1U<<bm_acc));
        for(uint Ei=0; Ei<ene_levels-i; ++Ei){
            uint Ej=Ei+i;
            for(uint k=0; k<ene_qbits; ++k){
                W_case_masks[i][Ei] |= ((Ei>>k & 1U) << bm_enes_old[k]) | ((Ej>>k & 1U) << bm_enes_new[k]);
            }
#if !defined(CUDA_HOST)
            host_W_case_masks[uptri_sum_formula(i)+Ei] = W_case_masks[i][Ei];
#else
            DEBUG_CALL(std::cout<<"W_case_masks["<<i<<"]["<<Ei<<"] = "<<W_case_masks[i][Ei]<<std::endl);
#endif
        }
    }
#if !defined(CUDA_HOST)
    cudaError_t err_code;
    err_code = cudaMemcpyToSymbol(dev_W_fs1, W_fs1.data(), ene_levels*sizeof(double), 0, cudaMemcpyHostToDevice);
    if(err_code!=cudaSuccess)
        printf("ERROR: cudaMemcpyToSymbol(), %s\n",cudaGetErrorString(err_code));

    err_code = cudaMemcpyToSymbol(dev_W_fs2, W_fs2.data(), ene_levels*sizeof(double), 0, cudaMemcpyHostToDevice);
    if(err_code!=cudaSuccess)
        printf("ERROR: cudaMemcpyToSymbol(), %s\n",cudaGetErrorString(err_code));

    
    err_code = cudaMalloc((void**)&dev_W_case_masks, W_case_masks_size*sizeof(uint)); 
    if(err_code!=cudaSuccess)
        printf("ERROR: cudaMalloc(), %s\n",cudaGetErrorString(err_code));
    err_code = cudaMemcpy(dev_W_case_masks, host_W_case_masks, W_case_masks_size*sizeof(uint), cudaMemcpyHostToDevice); 
    if(err_code!=cudaSuccess)
        printf("ERROR: cudaMemcpy(), %s\n",cudaGetErrorString(err_code));
#endif
}

uint creg_to_uint(const std::vector<uint>& c_reg){
    if(c_reg.size()<1)
        throw std::runtime_error("ERROR: size of register zero.");

    uint ret = c_reg[0];
    for(uint j=1U; j<c_reg.size(); ++j)
       ret += c_reg[j] << j; 

    return ret;
}
//XXX: the following two functions commented have parts define in suqa

void reset_non_state_qbits(ComplexVec& state){
#if defined(CUDA_HOST)
    DEBUG_CALL(std::cout<<"\n\nBefore reset"<<std::endl);
    DEBUG_CALL(sparse_print((double*)gState.data, gState.size()));
#endif
    std::vector<double> rgenerates(ene_qbits);

    for(auto& el : rgenerates) el = rangen.doub();
    suqa::apply_reset(state, bm_enes_old, rgenerates);

    for(auto& el : rgenerates) el = rangen.doub();
    suqa::apply_reset(state, bm_enes_new, rgenerates);

    suqa::apply_reset(state, bm_acc, rangen.doub());
#if defined(CUDA_HOST)
    DEBUG_CALL(std::cout<<"\n\nAfter reset"<<std::endl);
    DEBUG_CALL(sparse_print((double*)gState.data, gState.size()));
#endif
}


////TODO: can be optimized for multiple qbits measures?
void measure_qbits(ComplexVec& state, const std::vector<uint>& qs, std::vector<uint>& cs){
    for(uint k = 0U; k < qs.size(); ++k)
        suqa::measure_qbit(state, qs[k], cs[k], rangen.doub());
}


__global__ 
void kernel_qms_crm(Complex *const state, uint len, uint q_control, uint q_target, Complex rphase){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            state[i]*= rphase;
        }
        i+=gridDim.x*blockDim.x;
    }
}


void qms_crm(ComplexVec& state, const uint& q_control, const uint& q_target, const int& m){
    Complex rphase = (m>0) ? rphase_m[m] : rphase_m[-m];
    if(m<0) rphase.y*=-1;

#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        // for the swap, not only q_target:1 but also q_control:1
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            state[i]*= rphase;
        }
    }
#else // CUDA defined
    kernel_qms_crm<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q_control, q_target, rphase);
#endif
}

void qms_qft(ComplexVec& state, const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=qsize-1; outer_i>=0; outer_i--){
        suqa::apply_h(state, qact[outer_i]);
        for(int inner_i=outer_i-1; inner_i>=0; inner_i--){
            qms_crm(state, qact[inner_i], qact[outer_i], -1-(outer_i-inner_i));
        }
    }
}


void qms_qft_inverse(ComplexVec& state, const std::vector<uint>& qact){
    int qsize = qact.size();
    for(int outer_i=0; outer_i<qsize; outer_i++){
        for(int inner_i=0; inner_i<outer_i; inner_i++){
            qms_crm(state, qact[inner_i], qact[outer_i], 1+(outer_i-inner_i));
#if defined(CUDA_HOST)
            DEBUG_CALL(std::cout<<"In qms_qft_inverse() after crm: outer_i = "<<outer_i<<", inner_i = "<<inner_i<<std::endl);
            DEBUG_CALL(sparse_print((double*)state.data, state.size()));
#endif
        }
        suqa::apply_h(state, qact[outer_i]);
#if defined(CUDA_HOST)
            DEBUG_CALL(std::cout<<"In qms_qft_inverse() after apply_h: outer_i = "<<outer_i<<std::endl);
            DEBUG_CALL(sparse_print((double*)state.data, state.size()));
#endif
    }
}

void apply_phase_estimation(ComplexVec& state, const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(std::cout<<"apply_phase_estimation()"<<std::endl);
    suqa::apply_h(state,q_target);
#if defined(CUDA_HOST)
    DEBUG_CALL(std::cout<<"after qi_h(state,q_target)"<<std::endl);
    DEBUG_CALL(sparse_print((double*)state.data, state.size()));
#endif

    // apply CUs
    for(int trg = q_target.size() - 1; trg > -1; --trg){
        uint powr = (1U << (q_target.size()-1-trg));
        cevolution(state, powr*t, powr*n, q_target[trg], q_state);
    }
#if defined(CUDA_HOST)
    DEBUG_CALL(std::cout<<"\nafter evolutions"<<std::endl);
    DEBUG_CALL(sparse_print((double*)state.data, state.size()));
#endif
    
    // apply QFT^{-1}
    qms_qft_inverse(state, q_target); 

}

void apply_phase_estimation_inverse(ComplexVec& state, const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(std::cout<<"apply_phase_estimation_inverse()"<<std::endl);

    // apply QFT
    qms_qft(state, q_target); 


    // apply CUs
    for(uint trg = 0; trg < q_target.size(); ++trg){
        uint powr = (1U << (q_target.size()-1-trg));
        cevolution(state, -powr*t, powr*n, q_target[trg], q_state);
    }
    
    suqa::apply_h(state,q_target);
}


void apply_Phi_old(){

    apply_phase_estimation(gState, bm_states, bm_enes_old, t_phase_estimation, n_phase_estimation);

}

void apply_Phi_old_inverse(){

    apply_phase_estimation_inverse(gState, bm_states, bm_enes_old, t_phase_estimation, n_phase_estimation);

}

void apply_Phi(){

    apply_phase_estimation(gState, bm_states, bm_enes_new, t_phase_estimation, n_phase_estimation);

}

void apply_Phi_inverse(){

    apply_phase_estimation_inverse(gState, bm_states, bm_enes_new, t_phase_estimation, n_phase_estimation);

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

__global__
void kernel_qms_apply_W(Complex *const state, uint len, uint ene_levels, uint q_acc, uint W_mask, uint *const dev_W_case_masks){
    // call W_case_masks, W_fs1 and W_fs2 from constant memory
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        bool matching = false;
        uint dE;
        for(dE=1; dE<ene_levels; ++dE){
            for(uint k=0; k<(ene_levels-dE) && !matching; ++k){
                matching = ((i & W_mask) == dev_W_case_masks[uptri_sum_formula(dE)+k]);
            }
            if(matching)
                break;
        }
        if(matching){
            uint j = i & ~(1U << q_acc);
            suqa::apply_2x2mat_doub(state[j], state[i], dev_W_fs2[dE], dev_W_fs1[dE], dev_W_fs1[dE], -dev_W_fs2[dE]);
        }else if((i >> q_acc) & 1U){
            uint j = i & ~(1U << q_acc);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}

void apply_W(){
#if defined(CUDA_HOST)
    for(uint i = 0U; i < gState.size(); ++i){
        bool matching = false;
        uint dE;
        for(dE=1; dE<ene_levels; ++dE){
            for(uint k=0; k<W_case_masks[dE].size() && !matching; ++k){
                matching = ((i & W_mask) == W_case_masks[dE][k]);
            }
            if(matching)
                break;
        }
        if(matching){
            uint j = i & ~(1U << bm_acc);
            suqa::apply_2x2mat_doub(gState[j], gState[i], W_fs2[dE], W_fs1[dE], W_fs1[dE], -W_fs2[dE]);
        }else if((i >> bm_acc) & 1U){
            uint j = i & ~(1U << bm_acc);
            suqa::swap_cmpx(&gState[i],&gState[j]);
        }
    }
#else // CUDA defined
    //TODO: implement kernel
    qms::kernel_qms_apply_W<<<suqa::blocks,suqa::threads>>>(gState.data, gState.size(), ene_levels, bm_acc, W_mask, dev_W_case_masks);
#endif
}

void apply_W_inverse(){
    apply_W();
}

void apply_U(){
//    DEBUG_CALL(cout<<"\n\nApply U"<<endl);
    apply_C(gState, bm_states, gCi);
//    DEBUG_CALL(cout<<"\n\nAfter apply C = "<<gCi<<endl);
//    DEBUG_CALL(sparse_print(gState));




    apply_Phi();
//    DEBUG_CALL(cout<<"\n\nAfter second phase estimation"<<endl);
//    DEBUG_CALL(sparse_print(gState));



    apply_W();
//    DEBUG_CALL(cout<<"\n\nAfter apply W"<<endl);
//    DEBUG_CALL(sparse_print(gState));
}

void apply_U_inverse(){
    apply_W_inverse();
//    DEBUG_CALL(cout<<"\n\nAfter apply W inverse"<<endl);
//    DEBUG_CALL(sparse_print(gState));
    apply_Phi_inverse();
//    DEBUG_CALL(cout<<"\n\nAfter inverse second phase estimation"<<endl);
//    DEBUG_CALL(sparse_print(gState));
    apply_C_inverse(gState,bm_states,gCi);
//    DEBUG_CALL(cout<<"\n\nAfter apply C inverse = "<<gCi<<endl);
//    DEBUG_CALL(sparse_print(gState));
}


void init_measure_structs(){
    
    fill_meas_cache(bm_states, Xmatstem);
}

double measure_X(){
    if(Xmatstem==""){
        return 0.0;
    }

    std::vector<uint> classics(state_qbits);
    
    apply_measure_rotation(gState);

    measure_qbits(gState, bm_states, classics);

    apply_measure_antirotation(gState);

    uint meas = 0U;
    for(uint i=0; i<state_qbits; ++i){
        meas |= (classics[i] << i);
    }

    return get_meas_opvals(meas);
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
    reset_non_state_qbits(gState);
    DEBUG_CALL(std::cout<<"state after reset"<<std::endl);
    DEBUG_READ_STATE();
    apply_Phi_old();
    DEBUG_CALL(std::cout<<"\n\nAfter first phase estimation"<<std::endl);
    DEBUG_READ_STATE();

    gCi = draw_C();
    DEBUG_CALL(std::cout<<"\n\ndrawn C = "<<gCi<<std::endl);
    apply_U();


    suqa::measure_qbit(gState, bm_acc, c_acc, rangen.doub());

    if (c_acc == 1U){
        DEBUG_CALL(std::cout<<"accepted"<<std::endl);
        double Enew_meas_d;
        std::vector<uint> c_E_news(ene_qbits,0), c_E_olds(ene_qbits,0);
        DEBUG_CALL(std::cout<<"Measuring energy new"<<std::endl);
        measure_qbits(gState, bm_enes_new, c_E_news);
        DEBUG_CALL(double tmp_E=creg_to_uint(c_E_news)/(double)(t_PE_factor*ene_levels));
        DEBUG_CALL(std::cout<<"  energy measure : "<<tmp_E<<std::endl); 
        apply_Phi_inverse();
        if(take_measure){
            Enew_meas_d = creg_to_uint(c_E_news)/(double)(t_PE_factor*ene_levels);
            E_measures.push_back(Enew_meas_d);
            for(uint ei=0U; ei<ene_qbits; ++ei){
                suqa::apply_reset(gState, bm_enes_new[ei],rangen.doub());
            }
            X_measures.push_back(measure_X());
////            X_measures.push_back(0.0);
            DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
            DEBUG_CALL(std::cout<<"\n\nAfter X measure"<<std::endl);
            DEBUG_READ_STATE();
            DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
////           reset_non_state_qbits();
            for(uint ei=0U; ei<ene_qbits; ++ei)
                suqa::apply_reset(gState, bm_enes_new[ei],rangen.doub());
            apply_Phi();
            measure_qbits(gState, bm_enes_new, c_E_news);
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
        measure_qbits(gState, bm_enes_old, c_E_olds);
        Eold_meas = creg_to_uint(c_E_olds);
        Eold_meas_d = Eold_meas/(double)(t_PE_factor*ene_levels);
        measure_qbits(gState, bm_enes_new, c_E_news);
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
                suqa::apply_reset(gState, bm_enes_new[ei],rangen.doub());
                X_measures.push_back(measure_X());
                DEBUG_CALL(std::cout<<"\n\nAfter X measure"<<std::endl);
                DEBUG_READ_STATE();
                DEBUG_CALL(std::cout<<"  X measure : "<<X_measures.back()<<std::endl); 
            for(uint ei=0U; ei<ene_qbits; ++ei)
                suqa::apply_reset(gState, bm_enes_new[ei],rangen.doub());
                apply_Phi();
                measure_qbits(gState, bm_enes_new, c_E_news);
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
        suqa::measure_qbit(gState, bm_acc, c_acc_trash, rangen.doub()); 
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

    return ret;
}

void setup(double beta){
    qms::fill_rphase(qms::ene_qbits+1);
    qms::fill_bitmap();
    qms::fill_W_utils(beta, qms::t_PE_factor);
    if(qms::Xmatstem!="")
        qms::init_measure_structs();

#if !defined(CUDA_HOST) && !defined(NDEBUG)
    host_state = new Complex[qms::Dim];
#endif
}

void clear(){
#if !defined(CUDA_HOST)
    cudaFree(dev_W_case_masks);
#ifndef NDEBUG
    delete [] host_state;
#endif
#endif

}


}
