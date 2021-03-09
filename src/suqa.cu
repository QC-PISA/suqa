#include "suqa.cuh"

void suqa::print_banner(){
    printf("\n"
"                                                 \n" 
"   ███████╗██╗   ██╗ ██████╗  █████╗     \n" 
"   ██╔════╝██║   ██║██╔═══██╗██╔══██╗    \n" 
"   ███████╗██║   ██║██║   ██║███████║    \n" 
"   ╚════██║██║   ██║██║▄▄ ██║██╔══██║    \n" 
"   ███████║╚██████╔╝╚██████╔╝██║  ██║    \n" 
"   ╚══════╝ ╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═╝    \n" 
"                                          \n" 
"\n Simulator for Universal Quantum Algorithms\n\n");
}


ComplexVec suqa::state;
#ifdef SPARSE
std::vector<uint> suqa::actives;
#endif

#if !defined(NDEBUG) && defined(GPU)
double *host_state_re, *host_state_im;
#endif

// global control mask:
// it applies every next operation 
// using it as condition (the user should make sure
// to use it only for operations not involving it)
uint suqa::gc_mask;

uint suqa::nq;

#ifdef GPU
uint suqa::blocks, suqa::threads;
cudaStream_t suqa::stream1, suqa::stream2;
#endif

void suqa::activate_gc_mask(const bmReg& q_controls){
    for(const auto& q : q_controls)
        suqa::gc_mask |= 1U << q;
}

void suqa::deactivate_gc_mask(const bmReg& q_controls){
    for(const auto& q : q_controls)
        suqa::gc_mask &= ~(1U << q);
}

#ifdef GPU
#include "suqa_kernels.cuh"
double* host_partial_ret, *dev_partial_ret;
#else
#include "suqa_cpu.hpp"
#endif

void suqa::init_state() {
#ifdef GPU
    kernel_suqa_init_state<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size());
#else
    func_suqa_init_state(suqa::state.data);
#endif
}

double suqa::vnorm(){
#ifdef GPU
    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, suqa::state.data_re, suqa::state.data_im, suqa::state.size());
    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    double ret = 0.0;
    for(uint bid=0; bid<suqa::blocks; ++bid){
        
//        printf("host_partial_ret[%d(/%d)] = %.10lg\n",bid, suqa::blocks,host_partial_ret[bid]);
        ret += host_partial_ret[bid]; 
    } 
    return sqrt(ret);
#else
    return sqrt(func_suqa_vnorm(suqa::state.data));
#endif
}




void suqa::vnormalize(){
    double vec_norm = suqa::vnorm();
#ifndef NDEBUG
    std::cout<<"vec_norm = "<<vec_norm<<std::endl;
#endif

#ifdef GPU
    // using the inverse, since division is not built-in in cuda
    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads>>>(suqa::state.data, 2*suqa::state.size(),1./vec_norm);
//    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(v.data_re, v.size(),1./vec_norm);
//    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream2>>>(v.data_im, v.size(),1./vec_norm);
#else
    func_suqa_vnormalize_by(suqa::state.data,1./vec_norm);
#endif
}

//  X GATE



void suqa::apply_x(uint q){
#ifdef GPU
    kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#else
    func_suqa_x(suqa::state.data_re, suqa::state.data_im, q, suqa::gc_mask);
#endif
}

void suqa::apply_x(const bmReg& qs){
    for(const auto& q : qs)
		suqa::apply_x(q);
}  


//  Y GATE


void suqa::apply_y(uint q){
#ifdef GPU
    kernel_suqa_y<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#else
    func_suqa_y(suqa::state.data_re, suqa::state.data_im, q, suqa::gc_mask);
#endif
}  

void suqa::apply_y(const bmReg& qs){
    for (const auto& q : qs)
        suqa::apply_y(q);
}  

//  Z GATE

void suqa::apply_z(uint q){
    suqa::apply_u1(q, M_PI);
}  

void suqa::apply_z(const bmReg& qs){
    for(const auto& q : qs)
		suqa::apply_u1(q, M_PI);
}  


#ifdef GPU
//  SIGMA+ = 1/2(X+iY) GATE


void suqa::apply_sigma_plus(uint q){
#ifdef GPU
    kernel_suqa_sigma_plus<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#else 
    func_suqa_sigma_plus(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#endif
}  

void suqa::apply_sigma_plus(const bmReg& qs){
    for (const auto& q : qs)
        suqa::apply_sigma_plus(q);
}


//  SIGMA- = 1/2(X-iY) GATE


void suqa::apply_sigma_minus(uint q){
#ifdef GPU
    kernel_suqa_sigma_minus<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#else 
    func_suqa_sigma_minus(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#endif
}  

void suqa::apply_sigma_minus(const bmReg& qs){
    for(const auto& q : qs)
        suqa::apply_sigma_minus(q);
}

#endif

//  HADAMARD GATE



void suqa::apply_h(uint q){
#ifdef GPU
    kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, suqa::gc_mask);
#else
    func_suqa_h(suqa::state.data_re, suqa::state.data_im, q, suqa::gc_mask);
#endif
}  


void suqa::apply_h(const bmReg& qs){
    for(const auto& q : qs){
        suqa::apply_h(q);
    }
}  


//  PI/8 GATES

// T gate (single qubit)
void suqa::apply_t(uint q){
    suqa::apply_u1(q, M_PI / 4.0);
}  
// T gate (multiple qubits)
void suqa::apply_t(const bmReg& qs){
    for (const auto& q : qs) {
        suqa::apply_u1(q, M_PI / 4.0);
    }
}  

// T^{\dagger} gate (single qubit)
void suqa::apply_tdg(uint q){
    suqa::apply_u1(q, -M_PI / 4.0);
}  
// T^{\dagger} gate (multiple qubits)
void suqa::apply_tdg(const bmReg& qs){
    for(const auto& q : qs){
		suqa::apply_u1(q, -M_PI / 4.0);
    }
}  

// PI/4 GATES

// S gate (single qubit)
void suqa::apply_s(uint q){
    suqa::apply_u1(q, M_PI / 2.0);
}  
// S gate (multiple qubits)
void suqa::apply_s(const bmReg& qs){
    for (const auto& q : qs) {
        suqa::apply_u1(q, M_PI / 2.0);
    }
}  

// U1 GATE

void suqa::apply_u1(uint q, double phase){
	suqa::apply_u1(q, 1U, phase);
}

void suqa::apply_u1(uint q, uint q_mask, double phase){
    uint qmask = suqa::gc_mask|(q_mask<<q);
    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);
#ifdef GPU
    kernel_suqa_u1<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, phasec, qmask, suqa::gc_mask);
#else
    func_suqa_u1(state.data_re, state.data_im, q, phasec, qmask, suqa::gc_mask);
#endif
}


//  CONTROLLED-NOT GATE


void suqa::apply_cx(const uint& q_control, const uint& q_target, const uint& q_mask){
    uint mask_qs = suqa::gc_mask;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);
#ifdef GPU
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask|(1U<<q_target), mask_qs|(1U<<q_target), q_target);
#else
    func_suqa_mcx(suqa::state.data_re, suqa::state.data_im, mask, mask_qs, q_target);
#endif
}  

void suqa::apply_mcx(const bmReg& q_controls, const uint& q_target){
    uint mask = suqa::gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
#ifdef GPU
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask|(1U<<q_target), mask|(1U<<q_target), q_target);
#else
    func_suqa_mcx(suqa::state.data_re, suqa::state.data_im, mask, mask, q_target);
#endif
}  


void suqa::apply_mcx(const bmReg& q_controls, const bmReg& q_mask, const uint& q_target){
    uint mask = suqa::gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = suqa::gc_mask;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }
#ifdef GPU
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask|(1U<<q_target), mask|(1U<<q_target)_qs, q_target);
#else
    func_suqa_mcx(suqa::state.data_re, suqa::state.data_im, mask, mask_qs, q_target);
#endif
}  

void suqa::apply_cu1(uint q_control, uint q_target, double phase, uint q_mask){
    uint mask_qs = (1U << q_target) | suqa::gc_mask;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);

    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);
#ifdef GPU
    kernel_suqa_mcu1<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask, mask_qs, q_target, phasec);
#else
    func_suqa_mcu1(suqa::state.data_re, suqa::state.data_im, mask, mask_qs, phasec);
#endif
}

void suqa::apply_mcu1(const bmReg& q_controls, const bmReg& q_mask, const uint& q_target, double phase){
    uint mask = (1U << q_target) | suqa::gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = (1U << q_target) | suqa::gc_mask;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }

    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);

#ifdef GPU
    kernel_suqa_mcu1<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask, mask_qs, q_target, phasec);
#else
    func_suqa_mcu1(suqa::state.data_re, suqa::state.data_im, mask, mask_qs, phasec);
#endif
}

void suqa::apply_mcu1(const bmReg& q_controls, const uint& q_target, double phase) {
    suqa::apply_mcu1(q_controls, std::vector<uint>(q_controls.size(), 1U), q_target, phase);
}


void suqa::apply_swap(const uint& q1, const uint& q2){
    // swap gate: 00->00, 01->10, 10->01, 11->11
    // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
    uint mask00 = suqa::gc_mask;
    uint mask11 = mask00;
    uint mask_q1 = (1U << q1);
    uint mask_q2 = (1U << q2);
    mask11 |= mask_q1 | mask_q2;
#ifdef GPU
    kernel_suqa_swap<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask00, mask11, mask_q1, mask_q2);
#else
    func_suqa_swap(suqa::state.data_re, suqa::state.data_im, mask00, mask11, mask_q1, mask_q2);
#endif
}

//TODO: implement for cpu
#ifdef GPU
void suqa::apply_phase_list(uint q0, uint q_size, const std::vector<double>& phases){
    if(!(q_size>0U and (uint)phases.size()==(1U<<q_size))){
        throw std::runtime_error("ERROR: in suqa::apply_phase_list(): invalid q_size or phases.size()");
    }

    uint mask0s = suqa::gc_mask;
    uint size_mask = 1U; // 2^0
    for(uint i=1U; i<q_size; ++i){
        size_mask |= (1U << i); // 2^i
    }

    std::vector<Complex> c_phases(phases.size());
    for(uint i=0U; i<phases.size(); ++i){
        sincos(phases[i],&c_phases[i].y,&c_phases[i].x);
    }

#ifdef GPU
    HANDLE_CUDACALL(cudaMemcpyToSymbol(const_phase_list, c_phases.data(), phases.size()*sizeof(Complex), 0, cudaMemcpyHostToDevice));
    kernel_suqa_phase_list<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re,suqa::state.data_im,suqa::state.size(),mask0s,q0,size_mask);
#else
    func_suqa_phase_list(suqa::state.data_re,suqa::state.data_im,suqa::state.size(),c_phases,mask0s,q0,size_mask);
#endif
     
}
#endif // GPU

/* Pauli Tensor Product rotations */

// rotation by phase in the direction of a pauli tensor product
void suqa::apply_pauli_TP_rotation(const bmReg& q_apply, const std::vector<uint>& pauli_TPtype, double phase){
    uint mask0s = suqa::gc_mask;
    uint mask1s = mask0s;
    uint mask_q1, mask_q2, mask_q3;
    for(const auto& q : q_apply){
        mask1s |= (1U << q);
    }
    double sph, cph;
    sincos(phase, &sph, &cph);

    if(q_apply.size()!=pauli_TPtype.size()){
        throw std::runtime_error("ERROR: in suqa::apply_pauli_TP_rotation(): mismatch between qubits number and pauli types specified");
    }

    std::vector<uint> pauli_TPtype_cpy(pauli_TPtype);
    std::vector<uint> q_apply_cpy(q_apply);

    if(q_apply.size()==1U){
        mask_q1 = (1U << q_apply[0]);
#ifdef GPU
        // better to keep distinct kernels to minimize explicit branching in the device
        // (even if maybe the ptx compiler is smart enough to not cause overheads)
        switch(pauli_TPtype[0]){
            case PAULI_X:
                kernel_suqa_pauli_TP_rotation_x<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            case PAULI_Y:
                kernel_suqa_pauli_TP_rotation_y<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            case PAULI_Z:
                kernel_suqa_pauli_TP_rotation_z<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            default:
                break;
        }
#else
        func_suqa_pauli_TP_rotation_pauli1(pauli_TPtype[0],suqa::state.data_re, suqa::state.data_im, mask0s, mask1s, mask_q1, cph, sph);
#endif
    }else if(q_apply.size()==2U){
        if(pauli_TPtype_cpy[0]>pauli_TPtype_cpy[1]){ //sort cases
            std::swap(pauli_TPtype_cpy[0],pauli_TPtype_cpy[1]);
            std::swap(q_apply_cpy[0],q_apply_cpy[1]);
        }
        mask_q1 = (1U << q_apply_cpy[0]);
        mask_q2 = (1U << q_apply_cpy[1]);
#ifdef GPU
        if(pauli_TPtype_cpy[0]==PAULI_X and pauli_TPtype_cpy[1]==PAULI_X){
            kernel_suqa_pauli_TP_rotation_xx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, cph, sph);
        }else if(pauli_TPtype_cpy[0]==PAULI_Y and pauli_TPtype_cpy[1]==PAULI_Y){
            kernel_suqa_pauli_TP_rotation_yy<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, cph, sph);
        }else if(pauli_TPtype_cpy[0]==PAULI_Z and pauli_TPtype_cpy[1]==PAULI_Z){
            kernel_suqa_pauli_TP_rotation_zz<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, cph, sph);
        }else if(pauli_TPtype_cpy[0]==PAULI_X and pauli_TPtype_cpy[1]==PAULI_Y){
            kernel_suqa_pauli_TP_rotation_xy<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, cph, sph);
        }else if(pauli_TPtype_cpy[0]==PAULI_X and pauli_TPtype_cpy[1]==PAULI_Z){
            kernel_suqa_pauli_TP_rotation_zx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q2, mask_q1, cph, sph);
        }else if(pauli_TPtype_cpy[0]==PAULI_Y and pauli_TPtype_cpy[1]==PAULI_Z){
            kernel_suqa_pauli_TP_rotation_zy<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q2, mask_q1, cph, sph);
        }
#else
        func_suqa_pauli_TP_rotation_pauli2(pauli_TPtype_cpy[0],pauli_TPtype_cpy[1],suqa::state.data_re, suqa::state.data_im, mask0s, mask1s, mask_q1, mask_q2, cph, sph);
#endif

    }else if(q_apply.size()==3U){
        if(pauli_TPtype_cpy[0]>pauli_TPtype_cpy[1]){ //sort cases
            std::swap(pauli_TPtype_cpy[0],pauli_TPtype_cpy[1]);
            std::swap(q_apply_cpy[0],q_apply_cpy[1]);
        }
        if(pauli_TPtype_cpy[1]>pauli_TPtype_cpy[2]){ //sort cases
            std::swap(pauli_TPtype_cpy[1],pauli_TPtype_cpy[2]);
            std::swap(q_apply_cpy[1],q_apply_cpy[2]);
        }
        if(pauli_TPtype_cpy[0]>pauli_TPtype_cpy[1]){ //sort cases
            std::swap(pauli_TPtype_cpy[0],pauli_TPtype_cpy[1]);
            std::swap(q_apply_cpy[0],q_apply_cpy[1]);
        }

        int i_z = -1, i1, i2;
        if(pauli_TPtype_cpy[2]==PAULI_Z){
            i_z=2;
            i1=0;
            i2=1;
        }else{
            throw std::runtime_error("ERROR: unimplemented pauli TP rotation with 3 qubits in the selected configuration");
        }
        mask_q3 = (1U << q_apply_cpy[i_z]);
        mask_q1 = (1U << q_apply_cpy[i1]);
        mask_q2 = (1U << q_apply_cpy[i2]);

        if(pauli_TPtype_cpy[i1]!=pauli_TPtype_cpy[i2])
            throw std::runtime_error("ERROR: unimplemented pauli TP rotation with 3 qubits in the selected configuration");
        
#ifdef GPU
        if(pauli_TPtype_cpy[i1]==PAULI_X and pauli_TPtype_cpy[i2]==PAULI_X){
                kernel_suqa_pauli_TP_rotation_zxx<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
        }else if(pauli_TPtype_cpy[i1]==PAULI_Y and pauli_TPtype_cpy[i2]==PAULI_Y){
                kernel_suqa_pauli_TP_rotation_zyy<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
        }else if(pauli_TPtype_cpy[i1]==PAULI_Z and pauli_TPtype_cpy[i2]==PAULI_Z){
                kernel_suqa_pauli_TP_rotation_zzz<<<suqa::blocks,suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
        }
#else
        func_suqa_pauli_TP_rotation_pauli3(pauli_TPtype_cpy[2],pauli_TPtype_cpy[i1],pauli_TPtype_cpy[i2],suqa::state.data_re, suqa::state.data_im, mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
#endif
    }else{
        throw std::runtime_error(("ERROR: unimplemented pauli tensor product rotation with "+std::to_string(q_apply.size())+" qubits").c_str());
    }
}

/* End of Pauli Tensor Product rotations */

void set_ampl_to_zero(const uint& q, const uint& val){
#ifdef GPU
    kernel_suqa_set_ampl_to_zero<<<suqa::blocks, suqa::threads>>>(suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q, val);
#else
    func_suqa_set_ampl_to_zero(suqa::state.data_re, suqa::state.data_im, q, val);
#endif
}


void suqa::measure_qbit(uint q, uint& c, double rdoub){
    double prob1 = 0.0;
    c=0U;
#ifdef GPU
//    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data, 2*state.size(), q);
    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, suqa::state.data_re, suqa::state.data_im, suqa::state.size(), q);
//    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),stream2>>>(dev_partial_ret+blocks, state.data_im, state.size(), q);
//    cudaDeviceSynchronize();
    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(uint bid=0; bid<suqa::blocks && prob1<rdoub; ++bid){
        prob1 += host_partial_ret[bid]; 
    } 
#else
    prob1 = func_suqa_prob1(suqa::state.data_re, suqa::state.data_im, q);
#endif

    c = (uint)(rdoub < prob1); // prob1=1 -> c = 1 surely
    uint c_conj = c^1U; // 1U-c, since c=0U or 1U

    // set to 0 coeffs with bm_acc 1-c
    set_ampl_to_zero(q, c_conj);
    suqa::vnormalize();
}

////TODO: can be optimized for multiple qbits measures?
void suqa::measure_qbits(const bmReg& qs, std::vector<uint>& cs,const std::vector<double>& rdoubs){
    for(uint k = 0U; k < qs.size(); ++k)
        suqa::measure_qbit(qs[k], cs[k], rdoubs[k]);
}


//
//
void suqa::prob_filter(const bmReg& qs, const std::vector<uint>& q_mask, double &prob){
    prob = 0.0;
    uint mask_qs = 0U;
    for(const auto& q : qs)
        mask_qs |= 1U << q;
    uint mask = 0U;
    for(uint k = 0U; k < q_mask.size(); ++k){
        if(q_mask[k]) mask |= q_mask[k] << qs[k];
    }
#ifdef GPU
//    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data, 2*state.size(), q);
    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, suqa::state.data_re, suqa::state.data_im, suqa::state.size(), mask_qs, mask);
//    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret+suqa::blocks, state.data_im, state.size(), mask_qs, mask);
//    cudaDeviceSynchronize();
    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(uint bid=0; bid<suqa::blocks; ++bid){
        prob += host_partial_ret[bid]; 
    } 
#else
    prob=func_suqa_prob_filter(suqa::state.data_re, suqa::state.data_im, mask_qs, mask);
#endif
}

#ifdef GPU
// RESET = measure + classical cx
void suqa::apply_reset(uint q, double rdoub){
//    DEBUG_CALL(std::cout<<"Calling apply_reset() with q="<<q<<"and rdoub="<<rdoub<<std::endl);
    uint c;
    suqa::measure_qbit(q, c, rdoub);
    if(c){ // c==1U
        suqa::apply_x(q);
        // suqa::vnormalize(state); // normalization shoud be guaranteed by the measure
    }
}  

// fake reset
//void suqa::apply_reset(ComplexVec& state, uint q, double rdoub){
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i >> q) & 1U){ // checks q-th digit in i
//            uint j = i & ~(1U << q); // j has 0 on q-th digit
//            state[j]+=state[i];
//            state[i].x = 0.0;
//            state[i].y = 0.0;
//        }
//    }
//}

void suqa::apply_reset(const bmReg& qs, std::vector<double> rdoubs){
    // qs.size() == rdoubs.size()
    for(uint i=0; i<qs.size(); ++i){
        suqa::apply_reset(qs[i], rdoubs[i]); 
    } 
}
#endif


void suqa::deallocate_state(){
    if(state.data!=nullptr){
#ifdef GPU
        HANDLE_CUDACALL(cudaFree(state.data));
#else
        delete[] state.data;
        state.data = nullptr;
#endif
    }
    state.vecsize=0U;
}

void suqa::allocate_state(uint Dim){
    if(suqa::state.data!=nullptr or Dim!=suqa::state.vecsize)
        deallocate_state();


    state.vecsize = Dim; 
#ifdef GPU
    HANDLE_CUDACALL(cudaMalloc((void**)&(suqa::state.data), 2*suqa::state.vecsize*sizeof(double)));
#else
    suqa::state.data = new double[2 * suqa::state.vecsize];
#endif
    // allocate both using re as offset, and im as access pointer.
    suqa::state.data_re = suqa::state.data;
    suqa::state.data_im = suqa::state.data_re + suqa::state.vecsize;

    suqa::init_state();
}



void suqa::setup(uint num_qubits){
    suqa::nq = num_qubits;
    uint Dim = 1U << suqa::nq;

#ifdef GPU
    suqa::threads = NUM_THREADS;
    suqa::blocks = (Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>MAXBLOCKS) suqa::blocks=MAXBLOCKS;
    printf("blocks: %u, threads: %u\n",suqa::blocks, suqa::threads);

    cudaHostAlloc((void**)&host_partial_ret,suqa::blocks*sizeof(double),cudaHostAllocDefault);
    cudaMalloc((void**)&dev_partial_ret, suqa::blocks*sizeof(double));  
// the following are allocated only for library versions of reduce
//    cudaHostAlloc((void**)&ret_re_im,2*sizeof(double),cudaHostAllocDefault);
//    cudaMalloc((void**)&d_ret_re_im,2*sizeof(double));

    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_CUDACALL( cudaGetDevice( &whichDevice ) );
    HANDLE_CUDACALL( cudaGetDeviceProperties( &prop, whichDevice ) );

    HANDLE_CUDACALL( cudaStreamCreate( &suqa::stream1 ) );
    if (!prop.deviceOverlap) {
        DEBUG_CALL(printf( "Device will not handle overlaps, so no "
        "speed up from streams\n" ));
        suqa::stream2 = suqa::stream1;
    }else{
        HANDLE_CUDACALL( cudaStreamCreate( &suqa::stream2 ) );
    }

#if !defined(NDEBUG)
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_re,Dim*sizeof(double),cudaHostAllocDefault));
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_im,Dim*sizeof(double),cudaHostAllocDefault));
#endif



#endif // ifdef GPU

    allocate_state(Dim);
}

void suqa::clear(){
//    cudaFree(d_ret_re_im);
//    cudaFreeHost(ret_re_im);
#ifdef GPU
    cudaFree(dev_partial_ret); 
    cudaFreeHost(host_partial_ret);

#ifndef NDEBUG
    HANDLE_CUDACALL(cudaFreeHost(host_state_re));
    HANDLE_CUDACALL(cudaFreeHost(host_state_im));
#endif

    HANDLE_CUDACALL( cudaStreamDestroy( suqa::stream1 ) );
    if (suqa::stream1!=suqa::stream2) {
        HANDLE_CUDACALL( cudaStreamDestroy( suqa::stream2 ) );
    }
#endif
    deallocate_state();
}

//int main(){
//    std::cout<<"Suqa unit testing to be implemented"<<std::endl;
//    return 0;
//}

