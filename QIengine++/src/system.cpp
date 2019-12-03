#include <vector>
#include <complex>
#include <string>

const std::complex<double> iu(0,1);
typedef std::complex<double> Complex;

//void cevolution(std::vector<std::complex<double>>& state, const double& dt, const uint& q_control, const std::vector<uint>& qstate){
//
//    if(qstate.size()!=2)
//        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");
//
//    uint cmask = (1U << q_control);
//	uint mask = cmask;
//    for(const auto& qs : qstate){
//        mask |= (1U << qs);
//    }
//
//	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
//        if((i_0 & mask) == cmask){
//      
//            uint i_1 = i_0 | (1U << qstate[0]);
//            uint i_2 = i_0 | (1U << qstate[1]);
//            uint i_3 = i_1 | i_2;
//
//            std::complex<double> a_0 = state[i_0];
//            std::complex<double> a_1 = state[i_1];
//            std::complex<double> a_2 = state[i_2];
//            std::complex<double> a_3 = state[i_3];
//            
//            state[i_0] = exp(-dt*iu)*a_0;
//            state[i_1] = exp(-dt*iu)*(cos(dt)*a_1 -sin(dt)*iu*a_2);
//            state[i_2] = exp(-dt*iu)*(-sin(dt)*iu*a_1 + cos(dt)*a_2);
//            state[i_3] = exp(-dt*iu)*a_3;
//        }
//    }
//
//}

/* Quantum evolutor of the state */
void cevolution(std::vector<std::complex<double>>& state, const double& t, const int& n, const uint& q_control, const std::vector<uint>& qstate){

    (void)n; // Trotter not needed
    double dt = t;

    if(qstate.size()!=3)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");
     uint cmask = (1U << q_control);
    uint mask = cmask; // (1U << qstate[0]) | (1U << qstate[0])
     for(const auto& qs : qstate){
         mask |= (1U << qs);
     }

    for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
         if((i_0 & mask) == cmask){
       
             uint i_1 = i_0 | (1U << qstate[0]);
             uint i_2 = i_0 | (1U << qstate[1]);
             uint i_3 = i_1 | i_2;
             uint i_4 = i_0 | (1U << qstate[2]);
             uint i_5 = i_4 | i_1;
             uint i_6 = i_4 | i_2;
             uint i_7 = i_4 | i_3;


             Complex a_0 = state[i_0];
             Complex a_1 = state[i_1];
             Complex a_2 = state[i_2];
             Complex a_3 = state[i_3];
             Complex a_4 = state[i_4];
             Complex a_5 = state[i_5];
             Complex a_6 = state[i_6];
             Complex a_7 = state[i_7];

             double dtp = dt/4.; 
             // apply 1/.4 (Id +X2 X1)
             state[i_0] = exp(-dtp*iu)*(cos(dtp)*a_0 -sin(dtp)*iu*a_6);
             state[i_1] = exp(-dtp*iu)*(cos(dtp)*a_1 -sin(dtp)*iu*a_7);
             state[i_2] = exp(-dtp*iu)*(cos(dtp)*a_2 -sin(dtp)*iu*a_4);
             state[i_3] = exp(-dtp*iu)*(cos(dtp)*a_3 -sin(dtp)*iu*a_5);
             state[i_4] = exp(-dtp*iu)*(cos(dtp)*a_4 -sin(dtp)*iu*a_2);
             state[i_5] = exp(-dtp*iu)*(cos(dtp)*a_5 -sin(dtp)*iu*a_3);
             state[i_6] = exp(-dtp*iu)*(cos(dtp)*a_6 -sin(dtp)*iu*a_0);
             state[i_7] = exp(-dtp*iu)*(cos(dtp)*a_7 -sin(dtp)*iu*a_1);

             a_0 = state[i_0];
             a_1 = state[i_1];
             a_2 = state[i_2];
             a_3 = state[i_3];
             a_4 = state[i_4];
             a_5 = state[i_5];
             a_6 = state[i_6];
             a_7 = state[i_7];

             // apply 1/.4 (X2 X0)
             state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_5);
             state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_4);
             state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_7);
             state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_6);
             state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_1);
             state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_0);
             state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_3);
             state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_2);

             a_0 = state[i_0];
             a_1 = state[i_1];
             a_2 = state[i_2];
             a_3 = state[i_3];
             a_4 = state[i_4];
             a_5 = state[i_5];
             a_6 = state[i_6];
             a_7 = state[i_7];

             // apply 1/.4 (X1 X0)
             state[i_0] = (cos(dtp)*a_0 -sin(dtp)*iu*a_3);
             state[i_1] = (cos(dtp)*a_1 -sin(dtp)*iu*a_2);
             state[i_2] = (cos(dtp)*a_2 -sin(dtp)*iu*a_1);
             state[i_3] = (cos(dtp)*a_3 -sin(dtp)*iu*a_0);
             state[i_4] = (cos(dtp)*a_4 -sin(dtp)*iu*a_7);
             state[i_5] = (cos(dtp)*a_5 -sin(dtp)*iu*a_6);
             state[i_6] = (cos(dtp)*a_6 -sin(dtp)*iu*a_5);
             state[i_7] = (cos(dtp)*a_7 -sin(dtp)*iu*a_4);
         }
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


void fill_meas_cache(const std::vector<uint>& bm_states, const std::string opstem){
    state_levels = (uint)pow(2,bm_states.size());

    iss.resize(state_levels);
    ass.resize(state_levels);

    meas_opvals.resize(state_levels);
    SXmat.resize(state_levels,std::vector<Complex>(state_levels));

    FILE * fil_re = fopen((opstem+"_vecs_re").c_str(),"r"); 
    FILE * fil_im = fopen((opstem+"_vecs_im").c_str(),"r"); 
    FILE * fil_vals = fopen((opstem+"_vals").c_str(),"r"); 
    double tmp_re,tmp_im;
    for(uint i=0; i<state_levels; ++i){
        fscanf(fil_vals, "%lg",&meas_opvals[i]);
        for(uint j=0; j<state_levels; ++j){
            fscanf(fil_re, "%lg",&tmp_re);
            fscanf(fil_im, "%lg",&tmp_im);
            SXmat[i][j] = tmp_re+tmp_im*iu;
        }
        fscanf(fil_re, "\n");
        fscanf(fil_im, "\n");
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
 
void apply_measure_rotation(std::vector<Complex>& state){

	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & meas_mask) == 0U){
      
            for(uint lvl=0; lvl<state_levels; ++lvl){
                iss[lvl] = i_0 | meas_mask_combs[lvl];
                ass[lvl] = state[iss[lvl]];
            }

            for(uint r=0; r<state_levels; ++r){
                state[iss[r]]=0.0;
                for(uint c=0; c<state_levels; ++c){
                     state[iss[r]] += SXmat[r][c]*ass[c];
                }
            }
        }
    }
}

void apply_measure_antirotation(std::vector<Complex>& state){

	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & meas_mask) == 0U){
      
            for(uint lvl=0; lvl<state_levels; ++lvl){
                iss[lvl] = i_0 | meas_mask_combs[lvl];
                ass[lvl] = state[iss[lvl]];
            }

            for(uint r=0; r<state_levels; ++r){
                state[iss[r]]=0.0;
                for(uint c=0; c<state_levels; ++c){
                     state[iss[r]] += conj(SXmat[c][r])*ass[c];
                }
            }
        }
    }
}

/* Metropolis update step */
std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};
std::vector<double> get_C_weigthsums(){ return C_weigthsums; }

void qi_h(std::vector<Complex>& state, const uint& q);


void apply_C(std::vector<Complex>& state, const std::vector<uint>& bm_states, const uint &Ci){
    if(Ci==0U){
        qi_h(state,bm_states[0]);
    }else if(Ci==1U){
        qi_h(state,bm_states[1]);
    }else if(Ci==2U){
        qi_h(state,bm_states[2]);
    }else{
        throw std::runtime_error("Error!");
    }
}

void apply_C_inverse(std::vector<Complex>& state, const std::vector<uint>& bm_states, const uint &Ci){
    apply_C(state, bm_states, Ci);
}
