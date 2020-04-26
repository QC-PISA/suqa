#pragma once
#include "suqa.cuh"


__global__
void kernel_suqa_init_state(double* state_re, double* state_im, size_t len) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < len) {
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x * blockDim.x;
    }
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        printf("This\n");
        state_re[0] = 1.0;
        state_im[0] = 0.0;
    }
}



//TODO: optimize reduce
__global__
void kernel_suqa_vnorm(double *dev_partial_ret_ptr, double *v_re, double *v_im, size_t len){
    extern __shared__ double local_ret[];
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        tmpval = v_re[i]; 
        local_ret[threadIdx.x] +=  tmpval*tmpval;
        tmpval = v_im[i]; 
        local_ret[threadIdx.x] +=  tmpval*tmpval;
//        if(v_re[i]>0.0)
//            printf("%u %.16lg, %.16lg; loc_ret[%d] = %.16lg\n",i, v_re[i], v_im[i], threadIdx.x, local_ret[threadIdx.x]);
//        tmpval = v_comp[i+blockDim.x]; 
//        local_ret[threadIdx.x] +=  tmpval*tmpval;
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",threadIdx.x, local_ret[threadIdx.x]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            local_ret[threadIdx.x] += local_ret[threadIdx.x+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (threadIdx.x < 512) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (threadIdx.x < 256) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (threadIdx.x < 128) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (threadIdx.x <  64) { local_ret[threadIdx.x] += local_ret[threadIdx.x +  64]; } __syncthreads(); }
//
//    if(threadIdx.x<32){
//        if (blockDim.x >= 64) local_ret[threadIdx.x] += local_ret[threadIdx.x + 32];
//        if (blockDim.x >= 32) local_ret[threadIdx.x] += local_ret[threadIdx.x + 16];
//        if (blockDim.x >= 16) local_ret[threadIdx.x] += local_ret[threadIdx.x +  8];
//        if (blockDim.x >=  8) local_ret[threadIdx.x] += local_ret[threadIdx.x +  4];
//        if (blockDim.x >=  4) local_ret[threadIdx.x] += local_ret[threadIdx.x +  2];
//        if (blockDim.x >=  2) local_ret[threadIdx.x] += local_ret[threadIdx.x +  1];
//    }

    if(threadIdx.x==0){
        dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
//        printf("dev_partial_ret_ptr[%d] = %.16lg\n",blockIdx.x,dev_partial_ret_ptr[blockIdx.x]);
    }
}

//__launch_bounds__(128, 6)
__global__ void kernel_suqa_vnormalize_by(double *v_comp, size_t len, double value){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i < len){
        v_comp[i]*=value;
        i += gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_x(double *const state_re, double *const state_im, size_t len, uint q, uint glob_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U <<q);
    while(i<len){
        if((i & glob_mask) == glob_mask){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            double tmpval = state_re[i];
            state_re[i]=state_re[j];
            state_re[j]=tmpval;
            tmpval = state_im[i];
            state_im[i]=state_im[j];
            state_im[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_y(double *const state_re, double *const state_im, size_t len, uint q, uint glob_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U <<q);
    while(i<len){
        if((i & glob_mask) == glob_mask){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
// j=0, i=1; ap[0] = -i*a[1]; ap[1]=i*a[0]
            double tmpval = state_re[i];
            state_re[i]=-state_im[j];
	    state_im[j]=-tmpval;
            tmpval = state_im[i];
            state_im[i]=state_re[j];
            state_re[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_sigma_plus(double *const state_re, double *const state_im, size_t len, uint q, uint glob_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U <<q);
    while(i<len){
        if((i & glob_mask) == glob_mask){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
        	state_re[j]=state_re[i];
        	state_im[j]=state_im[i];
        	state_re[i]=0;
        	state_im[i]=0;
		}
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_sigma_minus(double *const state_re, double *const state_im, size_t len, uint q, uint glob_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U <<q);
    while(i<len){
        if((i & glob_mask) == glob_mask){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
        	state_re[i]=state_re[j];
        	state_im[i]=state_im[j];
        	state_re[j]=0;
        	state_im[j]=0;
		}
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_h(double *state_re, double *state_im, size_t len, uint q, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    
    uint loc_mask = glob_mask | (1U << q);
    while(i_0<len){
        if((i_0 & loc_mask) == glob_mask){
            const uint i_1 = i_0 | (1U << q);
            double a_0_re = state_re[i_0];
            double a_1_re = state_re[i_1];
            double a_0_im = state_im[i_0];
            double a_1_im = state_im[i_1];
            
            state_re[i_0]= TWOSQINV*(a_0_re+a_1_re);
            state_re[i_1]= TWOSQINV*(a_0_re-a_1_re);
            state_im[i_0]= TWOSQINV*(a_0_im+a_1_im);
            state_im[i_1]= TWOSQINV*(a_0_im-a_1_im);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_u1(double *state_re, double *state_im, size_t len, uint q, Complex phase, uint qmask, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U << q);
    while(i<len){
        if((i & glob_mask) == qmask){  // q_mask 
            double tmpval = state_re[i]; 
            state_re[i] = state_re[i]*phase.x-state_im[i]*phase.y;
            state_im[i] = tmpval*phase.y+state_im[i]*phase.x;

        }
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_mcx(double *const state_re, double *const state_im, size_t len, uint control_mask, uint mask_qs, uint q_target){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & control_mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i];
            state_re[i]=state_re[j];
            state_re[j]=tmpval;
            tmpval = state_im[i];
            state_im[i]=state_im[j];
            state_im[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_mcu1(double *const state_re, double *const state_im, size_t len, uint control_mask, uint mask_qs, uint q_target, Complex rphase){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & control_mask) == mask_qs){
//            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i]; 
            state_re[i] = state_re[i]*rphase.x-state_im[i]*rphase.y;
            state_im[i] = tmpval*rphase.y+state_im[i]*rphase.x;
        }
        i+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_swap(double *const state_re, double *const state_im, size_t len, uint mask00, uint mask11, uint mask_q1, uint mask_q2){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & mask11) == mask00){
            // i -> ...00..., i_1 -> ...10..., i_2 -> ...01...
            uint i_1 = i | mask_q1;
            uint i_2 = i | mask_q2;
            double tmpval = state_re[i_1];
            state_re[i_1]=state_re[i_2];
            state_re[i_2]=tmpval;
            tmpval = state_im[i_1];
            state_im[i_1]=state_im[i_2];
            state_im[i_2]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

#define MAX_PHASES_NUM 128
__constant__ Complex const_phase_list[MAX_PHASES_NUM];
//supports up to PHASES_NUM complex phases

__global__ 
void kernel_suqa_phase_list(double *const state_re, double *const state_im, size_t len, uint mask0s, uint bm_offset, uint size_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if(i & mask0s){ // any state with gmask set
            uint ph_idx=(i>>bm_offset) & size_mask; //index in the phases list
            Complex cph = const_phase_list[ph_idx];
            double tmpval = state_re[i]; 
            state_re[i] = state_re[i]*cph.x-state_im[i]*cph.y;
            state_im[i] = tmpval*cph.y+state_im[i]*cph.x;
        }
        i+=gridDim.x*blockDim.x;
    }
}

__device__ __inline__
void util_rotate4(double *a, double *b, double *c, double *d, double ctheta, double stheta){
    // utility function for pauli rotation
    double cpy = *a;
    *a = cpy*ctheta -(*b)*stheta;
    *b = (*b)*ctheta + cpy*stheta;
    cpy = *c;
    *c = cpy*ctheta -(*d)*stheta;
    *d = (*d)*ctheta + cpy*stheta;
}


__global__
void kernel_suqa_pauli_TP_rotation_x(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;

            util_rotate4(&state_re[i_0],&state_im[i_1],&state_re[i_1],&state_im[i_0],ctheta,stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_y(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;

            util_rotate4(&state_re[i_0],&state_re[i_1],&state_im[i_0],&state_im[i_1],ctheta,-stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_z(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;

            util_rotate4(&state_re[i_0],&state_im[i_0],&state_im[i_1],&state_re[i_1],ctheta,stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_xx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            
            // 0<->3
            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,stheta);

            // 1<->2
            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_yy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;

            // 0<->3 yy is real negative on 0,3
            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,-stheta);

            // 1<->2 yy is real positive on 1,2
            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zz(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            
            // kl -> i(-)^(k+l) kl

            // +i
            util_rotate4(&state_re[i_0],&state_im[i_0],&state_re[i_3],&state_im[i_3],ctheta,stheta);

            // -i
            util_rotate4(&state_re[i_1],&state_im[i_1],&state_re[i_2],&state_im[i_2],ctheta,-stheta);

        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_xy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            
            // 0<->3
            util_rotate4(&state_re[i_0],&state_re[i_3],&state_im[i_0],&state_im[i_3],ctheta,-stheta);

            // 2<->1
            util_rotate4(&state_re[i_2],&state_re[i_1],&state_im[i_2],&state_im[i_1],ctheta,-stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            
            // ix on 0<->1
            util_rotate4(&state_re[i_0],&state_im[i_1],&state_re[i_1],&state_im[i_0],ctheta,stheta);

            // -ix on 2<->3
            util_rotate4(&state_re[i_2],&state_im[i_3],&state_re[i_3],&state_im[i_2],ctheta,-stheta);

        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            
            // iy on 0<->1
            util_rotate4(&state_re[i_0],&state_re[i_1],&state_im[i_0],&state_im[i_1],ctheta,-stheta);

            // -iy on 2<->3
            util_rotate4(&state_re[i_2],&state_re[i_3],&state_im[i_2],&state_im[i_3],ctheta,stheta);

        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zxx(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            uint i_4 = i_0 | mask_q3;
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;

            // 0<->3
            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,stheta);

            // 1<->2
            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);

            // 4<->7
            util_rotate4(&state_re[i_4],&state_im[i_7],&state_re[i_7],&state_im[i_4],ctheta,-stheta);

            // 5<->6
            util_rotate4(&state_re[i_5],&state_im[i_6],&state_re[i_6],&state_im[i_5],ctheta,-stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zyy(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            uint i_4 = i_0 | mask_q3;
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;


            // 0<->3 zyy is real negative on 0,3
            util_rotate4(&state_re[i_0],&state_im[i_3],&state_re[i_3],&state_im[i_0],ctheta,-stheta);

            // 1<->2 zyy is real positive on 1,2
            util_rotate4(&state_re[i_1],&state_im[i_2],&state_re[i_2],&state_im[i_1],ctheta,stheta);

            // 4<->7 zyy is real positive on 4,7
            util_rotate4(&state_re[i_4],&state_im[i_7],&state_re[i_7],&state_im[i_4],ctheta,stheta);

            // 5<->6 zyy is real negative on 5,6
            util_rotate4(&state_re[i_5],&state_im[i_6],&state_re[i_6],&state_im[i_5],ctheta,-stheta);
            
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zzz(double *const state_re, double *const state_im, size_t len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            uint i_4 = i_0 | mask_q3;
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;

            // +i on 0, 3, 5, 6
            util_rotate4(&state_re[i_0],&state_im[i_0],&state_re[i_3],&state_im[i_3],ctheta,stheta);
            util_rotate4(&state_re[i_5],&state_im[i_5],&state_re[i_6],&state_im[i_6],ctheta,stheta);

            // -i on 1, 2, 4, 7
            util_rotate4(&state_re[i_1],&state_im[i_1],&state_re[i_2],&state_im[i_2],ctheta,-stheta);
            util_rotate4(&state_re[i_4],&state_im[i_4],&state_re[i_7],&state_im[i_7],ctheta,-stheta);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

// sets amplitudes with value <val> in qubit <q> to zero
// !! it leaves the state unnormalized !!
__global__ void kernel_suqa_set_ampl_to_zero(double *state_re, double *state_im, size_t len, uint q, uint val){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i<len){
        if(((i >> q) & 1U) == val){
            state_re[i] = 0.0;
            state_im[i] = 0.0;
        }
        i += gridDim.x*blockDim.x;
    }
}


__global__ void kernel_suqa_prob1(double *dev_partial_ret_ptr, double *v_re, double *v_im, size_t len, uint q){
    extern __shared__ double local_ret[];
    uint tid = threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        if(i & (1U << q)){
            tmpval = v_re[i];
            local_ret[tid] +=  tmpval*tmpval;
            tmpval = v_im[i];
            local_ret[tid] +=  tmpval*tmpval;
        }
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",tid, local_ret[tid]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (tid < 512) { local_ret[tid] += local_ret[tid + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (tid < 256) { local_ret[tid] += local_ret[tid + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (tid < 128) { local_ret[tid] += local_ret[tid + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (tid <  64) { local_ret[tid] += local_ret[tid +  64]; } __syncthreads(); }
//
//    if(tid<32){
//        if (blockDim.x >= 64) local_ret[tid] += local_ret[tid + 32];
//        if (blockDim.x >= 32) local_ret[tid] += local_ret[tid + 16];
//        if (blockDim.x >= 16) local_ret[tid] += local_ret[tid +  8];
//        if (blockDim.x >=  8) local_ret[tid] += local_ret[tid +  4];
//        if (blockDim.x >=  4) local_ret[tid] += local_ret[tid +  2];
//        if (blockDim.x >=  2) local_ret[tid] += local_ret[tid +  1];
//    }

    if(tid==0) dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
}

__global__ void kernel_suqa_prob_filter(double *dev_partial_ret_ptr, double *v_re, double *v_im, size_t len, uint mask_qs, uint mask){
    extern __shared__ double local_ret[];
    uint tid = threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        if((i & mask_qs) == mask){
            tmpval = v_re[i];
            local_ret[tid] +=  tmpval*tmpval;
            tmpval = v_im[i];
            local_ret[tid] +=  tmpval*tmpval;
        }
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",tid, local_ret[tid]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (tid < 512) { local_ret[tid] += local_ret[tid + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (tid < 256) { local_ret[tid] += local_ret[tid + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (tid < 128) { local_ret[tid] += local_ret[tid + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (tid <  64) { local_ret[tid] += local_ret[tid +  64]; } __syncthreads(); }
//
//    if(tid<32){
//        if (blockDim.x >= 64) local_ret[tid] += local_ret[tid + 32];
//        if (blockDim.x >= 32) local_ret[tid] += local_ret[tid + 16];
//        if (blockDim.x >= 16) local_ret[tid] += local_ret[tid +  8];
//        if (blockDim.x >=  8) local_ret[tid] += local_ret[tid +  4];
//        if (blockDim.x >=  4) local_ret[tid] += local_ret[tid +  2];
//        if (blockDim.x >=  2) local_ret[tid] += local_ret[tid +  1];
//    }

    if(tid==0) dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
}


