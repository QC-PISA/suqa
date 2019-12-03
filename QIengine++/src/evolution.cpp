#include <vector>
#include <complex>

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

 






