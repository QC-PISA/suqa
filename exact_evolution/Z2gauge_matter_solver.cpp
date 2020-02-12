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


double H_mass_ferm(vector<int>& bin_state_i, vector<int>& bin_state_j, double mass){
    vector<int> ferm_part_i(4);
    vector<int> ferm_part_j(4);


    for(int i=0; i<4; ++i){
        ferm_part_i[i] = bin_state_i[i];
        ferm_part_j[i] = bin_state_j[i];
    }

    if(ferm_part_i != ferm_part_j){
        return 0.0;
    } else if(bin_to_dec(ferm_part_i, 4)%3==0){
        return 0.0;
    } else if((bin_to_dec(ferm_part_i, 4)-1)%3==0 && bin_to_dec(ferm_part_i, 4) != 10){
        return mass;
    } else if((bin_to_dec(ferm_part_i, 4)+1)%3==0 && bin_to_dec(ferm_part_i, 4) != 5){
        return -mass;
    } else if(bin_to_dec(ferm_part_i, 4) == 10){
        return -2*mass;
    } else{
        return 2*mass;
    }
}


int main(int argc, char ** argv){
    
    cout<<"\n\nZ2 gauge theory with staggered fermions. 3 links and 4 fermions.\n"<<endl;

    if(argc<1){
        cout<<"MASSA\n"<<endl;
    }

    double mass = stod(argv[1]);


    vector<int> bin_state(dimstate);
    vector<int> bin_state2(dimstate);

    int dec_state = 32;

    dec_to_bin(dec_state, bin_state, dimstate);
    dec_to_bin(dec_state, bin_state2, dimstate);
    printf("dec = %d\n", dec_state);
    
    
    for(int i=0; i<dimstate; ++i){
        printf("%d", bin_state[i]);     
    }
    printf("\n");
    
    dec_state = bin_to_dec(bin_state, dimstate);
    printf("decimal again = %d\n", dec_state);

    double a;
    a = 3;
    for(int b=0;b<(int)pow(2,7); ++b){
        dec_to_bin(b, bin_state, dimstate);
        dec_to_bin(b, bin_state2, dimstate);
        a=H_mass_ferm(bin_state, bin_state2, mass);
        printf("stato %d termine %.12lg\n", b, a);
    }
    return 0;
}
