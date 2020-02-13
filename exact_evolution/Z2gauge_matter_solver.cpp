#include <iostream>
#include <fstream>
#include <cstdio>
#include <complex>
#include <string>
#include <cstring>
#include <vector>
#include <omp.h>
#define ARMA_USE_LAPACK
#define ARMA_USE_OPENMP
#define ARMA_USE_SUPERLU
#include <armadillo>

#define Dim 1<<7
#define K1 dim1
#define K2 (dim1*K1)
#define K3 (dim1*K2)

#define DONE "\033[1;32m[DONE]\033[0m"

using namespace std;
using namespace arma;
typedef complex<double> Complex;
const Complex iu(0, 1);
const int dimstate = 7;
const int num_ferm = 4;
const int num_link = 3;



wall_clock timer, tglob;

void dec_to_bin(int dec_state, vector<int>& bin_state, int dimstate){
    for(int i=0;i<dimstate; ++i){
        bin_state[dimstate-i-1] = dec_state % 2;
        dec_state /= 2;
    }
}


int bin_to_dec(vector<int>& bin_state, int dimstate){
    int dec_state=0;
    for(int i=0;i<dimstate; ++i){
        dec_state += bin_state[dimstate-i-1]*(int)pow(2,i);
//        printf("partial = %d %d\n", (int) pow(2,i), bin_state[dimstate-i]); 
    }
    return dec_state;
}

//compute the element <i|H_m|j>
double H_mass_ferm(vector<int>& bin_state_i, vector<int>& bin_state_j, double mass){
    vector<int> ferm_part_i(4);
    vector<int> ferm_part_j(4);

 
    for(int i=0; i<4; ++i){
        ferm_part_i[i] = bin_state_i[i];
        ferm_part_j[i] = bin_state_j[i];
    }

    if(ferm_part_i != ferm_part_j){
        return 0.0;
    } else if(bin_to_dec(ferm_part_i, num_ferm)%3==0){
        return 0.0;
    } else if((bin_to_dec(ferm_part_i, num_ferm)-1)%3==0 && bin_to_dec(ferm_part_i, num_ferm) != 10){
        return -mass;
    } else if((bin_to_dec(ferm_part_i, num_ferm)+1)%3==0 && bin_to_dec(ferm_part_i, num_ferm) != 5){
        return mass;
    } else if(bin_to_dec(ferm_part_i, num_ferm) == 10){
        return 2*mass;
    } else{
        return -2*mass;
    }
}


//compute the element <j|H_g|i>
double H_gauge_link(vector<int>& bin_state_j, vector<int>& bin_state_i){
    vector<int> link_part_i(num_link);
    vector<int> link_part_j(num_link);

    for(int i=0; i<num_link; ++i){
       link_part_i[i] = bin_state_i[i+num_ferm]; 
       link_part_j[i] = bin_state_j[i+num_ferm]; 
    }

    if(link_part_i == link_part_j){
        return 0.0;
    } else{
        for(int i=0;i<num_link;++i){
            if(link_part_i[i]==0){
                link_part_i[i] = 1;
                if(link_part_i == link_part_j){
                    link_part_i[i]=0;
                    return 1.0;
                } else{
                    link_part_i[i]=0;
                }
            }
            if(link_part_i[i]==1){
                link_part_i[i] = 0;
                if(link_part_i == link_part_j){
                    link_part_i[i]=1;
                    return 1.0;
                } else{
                    link_part_i[i]=1;
                }
            }
           
        }
        return 0.0;
    }
}

double H_hop_X(vector<int>& bin_state_j, vector<int> bin_state_i){
    vector<int> link_part_i(num_link);
    vector<int> link_part_j(num_link);
    vector<int> ferm_part_i(num_ferm);
    vector<int> ferm_part_j(num_ferm);

    for(int i=0; i<num_link; ++i){
       link_part_i[i] = bin_state_i[i+num_ferm]; 
       link_part_j[i] = bin_state_j[i+num_ferm]; 
    }

  
    for(int i=0; i<num_ferm; ++i){
        ferm_part_i[i] = bin_state_i[i];
        ferm_part_j[i] = bin_state_j[i];
    }

    if(link_part_i != link_part_j){
        return 0.0;
    } else{ 
        if(ferm_part_i == ferm_part_j){
            return 0.0;
        } else {
            for(int i=0; i<num_link;++i){
                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
                
                if(ferm_part_i == ferm_part_j){
                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;

                return pow(-1, i+1)*0.25;
                } else{
                    ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                    ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
                }
            }
            return 0.0;
        }
    }
}


double H_hop_Y(vector<int>& bin_state_j, vector<int> bin_state_i){
    vector<int> link_part_i(num_link);
    vector<int> link_part_j(num_link);
    vector<int> ferm_part_i(num_ferm);
    vector<int> ferm_part_j(num_ferm);

    int sign = 1;

    for(int i=0; i<num_link; ++i){
       link_part_i[i] = bin_state_i[i+num_ferm]; 
       link_part_j[i] = bin_state_j[i+num_ferm]; 
    }

  
    for(int i=0; i<num_ferm; ++i){
        ferm_part_i[i] = bin_state_i[i];
        ferm_part_j[i] = bin_state_j[i];
    }

    if(link_part_i != link_part_j){
        return 0.0;
    } else{ 
        if(ferm_part_i == ferm_part_j){
            return 0.0;
        } else {
            for(int i=0; i<num_link;++i){
                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                if(ferm_part_i[i]==0) sign *= -1;
                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
                if(ferm_part_i[i]==0) sign *= -1;
                
                if(ferm_part_i == ferm_part_j){
                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;

                return sign*pow(-1, i+1)*0.25;
                } else{
                    ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                    ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
                    sign = 1;
                }
            }
            return 0.0;
        }
    }
}


     
int main(int argc, char ** argv){
    
    cout<<"\n\nZ2 gauge theory with staggered fermions. 3 links and 4 fermions.\n"<<endl;

    if(argc<5){
        cout<<"Generate H matrix for a Z2 gauge theory plus staggered fermions and compute the evolution of the -number density- of the leftmost fermion. The lattice is 1D and there are 4 staggered fermions and 3 gauge links. The initial state is the one with all 0s projected on the gauge invariant subspace\n\n";
        cout<<"usage:\n\n\t./z2_gauge_matter_solver <mass_parameter> <stepsize> <number of total steps> <name of generated file>"<<endl;
        exit(1);
    }

    double mass = stod(argv[1]);
    double dt = stod(argv[2]);
    int n = stod(argv[3]);
    string filestem = argv[4];

    vector<int> bin_state(dimstate);
    vector<int> bin_state2(dimstate);

    vec eigvals;
    mat eigvecs;

    string eigvalscachename = filestem+"_vals_cached";
    string eigvecscachename = filestem+"_vecs_cached";
    if(not ifstream(eigvalscachename.c_str()).good()){
        
/*---------------------Building the Matrix-------------*/

        cout<<"Construct the matrix ..."<<flush;      
        tglob.tic();
        
        mat H_mass(dimstate, dimstate);
        for(int a=0;a<i(int) pow(2,dimstate);++i){
            for(int b=0;b<(int) pow(2,dimstate);++i){
                
                H_mass(b, a) = H_mass_ferm()
            }   
        }

    }



//    dec_to_bin(dec_state, bin_state, dimstate);
//    dec_to_bin(dec_state, bin_state2, dimstate);
//    printf("dec = %d\n", dec_state);
//    
//    for(int i=0; i<dimstate; ++i){
//        printf("%d", bin_state[i]);     
//    }
//    printf("\n");
//    
//    dec_state = bin_to_dec(bin_state, dimstate);
//    printf("decimal again = %d\n", dec_state);
//
//    //for(int b=0;b<(int)pow(2,7); ++b){
//    //    dec_to_bin(b, bin_state, dimstate);
//    //    dec_to_bin(b, bin_state2, dimstate);
//    //    a=H_mass_ferm(bin_state, bin_state2, mass);
//    //    printf("stato %d termine %.12lg\n", b, a);
//    //}
//   
//    int a, b; 
//    double ris=0;
//    double ris2=0;
//    for(a=0;a<(int)pow(2,7);++a){
//        for(b=0;b<(int)pow(2,7); ++b){
//            dec_to_bin(a, bin_state, dimstate);
//            dec_to_bin(b, bin_state2, dimstate);
//            ris=H_hop_X(bin_state, bin_state2);
//            ris2=H_hop_Y(bin_state2, bin_state);
//           // printf("IL valore è %d %d %.12lf\n",a, b, ris);
//            if(ris2 != ris){
//                printf("NOT SYMMETRIC! %d %d %.12lf %.12lf\n\n", a, b, ris, ris2);
//            }
//        }
//    }
    return 0;
}
