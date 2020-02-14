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

double Apply_HhopX_site(vector<int>& state_in, vector<int>& state_out, int site, double amp){
    state_out[num_ferm-site] = (state_in[num_ferm-site]+1)%2;
    state_out[num_ferm-site-1] = (state_in[num_ferm-site-1]+1)%2;

    amp = 0.25*pow(-1, site);
    if(state_in[dimstate-site]==1){
        amp = -amp;
    }
    return amp;

}



double H_hop_X(vector<int>& bin_state_j, vector<int> bin_state_i){
      vector<int> state_aux(dimstate);
      double amp=0;

      for(int site=1; site<4; ++site){
        Apply_HhopX_site(bin_state_i, state_aux, site, amp);
        if(bin_state_j == state_aux){
            return amp;
        }        
      }
      return 0.0;

//    vector<int> link_part_i(num_link);
//    vector<int> link_part_j(num_link);
//    vector<int> ferm_part_i(num_ferm);
//    vector<int> ferm_part_j(num_ferm);
//
//    for(int i=0; i<num_link; ++i){
//       link_part_i[i] = bin_state_i[i+num_ferm]; 
//       link_part_j[i] = bin_state_j[i+num_ferm]; 
//    }
//
//  
//    for(int i=0; i<num_ferm; ++i){
//        ferm_part_i[i] = bin_state_i[i];
//        ferm_part_j[i] = bin_state_j[i];
//    }
//
//    if(link_part_i != link_part_j){
//        return 0.0;
//    } else if(ferm_part_i == ferm_part_j){
//            return 0.0;
//    } else {
//        for(int i=0; i<num_link;++i){
//            ferm_part_i[i] = (ferm_part_i[i]+1)%2;
//            ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
//            
//            if(ferm_part_i == ferm_part_j){
//            ferm_part_i[i] = (ferm_part_i[i]+1)%2;
//            ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
//            if(link_part_i[i]==1){
//                return -pow(-1, i)*0.25;
//            } else{
//                return pow(-1,i)*0.25;
//            }
//            } else{
//                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
//                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
//            }
//        }
//            return 0.0;
//    }
//    
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
    } else if(ferm_part_i == ferm_part_j){
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
            if(link_part_i[i]==1){
                sign*=-1;
                return sign*pow(-1, i)*0.25;
            } else{
                return sign*pow(-1, i);
            }
            } else{
                ferm_part_i[i] = (ferm_part_i[i]+1)%2;
                ferm_part_i[i+1] = (ferm_part_i[i+1]+1)%2;
                sign = 1;
            }
        }
            return 0.0;
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

    vector<int> a_state(dimstate);
    vector<int> b_state(dimstate);

    vec eigvals;
    mat eigvecs;

    string eigvalscachename = filestem+"_vals_cached";
    string eigvecscachename = filestem+"_vecs_cached";
    if(not ifstream(eigvalscachename.c_str()).good()){
        
/*---------------------Building the Matrix-------------*/

        cout<<"Construct the matrix ..."<<flush;      
        tglob.tic();
        
        mat H((int)pow(2,dimstate), (int)pow(2,dimstate));
        for(int a=0;a<(int) pow(2,dimstate);++a){
            for(int b=0;b<(int) pow(2,dimstate);++b){
                dec_to_bin(a, a_state, dimstate);
                dec_to_bin(b, b_state, dimstate);
                H(b,a) = H_mass_ferm(b_state, a_state, mass);
                H(b,a) += H_gauge_link(b_state, a_state);
                H(b,a) += H_hop_X(b_state, a_state);
                H(b,a) += H_hop_Y(b_state, a_state);
            }   
        }
        
        tglob.toc();
        cout<<"\b\b\b"<<DONE " in "<<tglob.toc()<<" seconds."<<endl;

        cout<<"H diagonalization ..."<<flush;
        tglob.tic();

        eig_sym(eigvals, eigvecs, H);

        tglob.toc();
        cout<<"\b\b\b"<<DONE<<" in "<<tglob.toc()<<" seconds."<<endl;

        // cache decomposed H matrix
        
        eigvals.save(eigvalscachename);
        eigvecs.save(eigvecscachename);
    }else{
        cout<<"\033[7;33mUsing cached eigenstuff\033[0m"<<endl;
        eigvals.load(eigvalscachename);
        eigvecs.load(eigvecscachename);
    }    
    
/*-------------- Evolution -----------*/
    cout<<"Evolution ..."<<flush;
    tglob.tic();
    // initialize state to all links the identity (state |0> using lexycographic order).
    int Dim=(int)pow(2,dimstate);
    vector<Complex> psi0(Dim,0.0);
    psi0[24]={1.0/sqrt(8),0.0};
    psi0[25]={-1.0/sqrt(8),0.0};
    psi0[26]={1.0/sqrt(8),0.0};
    psi0[27]={-1.0/sqrt(8),0.0};
    psi0[28]={1.0/sqrt(8),0.0};
    psi0[29]={-1.0/sqrt(8),0.0};
    psi0[30]={1.0/sqrt(8),0.0};
    psi0[31]={-1.0/sqrt(8),0.0};

    // initialize auxiliary buffer for the evolution and pointers
    vector<Complex> psi0_mom(Dim,0.0);
    int i, L, Lp;
    double t = 0.0;
    double plaq_tmp;
    vector<double> plaq(n+1,0.0);
    vector<int> bin_state(dimstate);
    //    #pragma omp parallel for collapse(2)
    for(Lp = 0; Lp < Dim; ++Lp) for(L = 0; L < Dim; ++L){
        psi0_mom[Lp] += eigvecs(L,Lp)*psi0[L];
    }    
    // compute plaquette on the state
    plaq_tmp=0.0;
//    #pragma omp parallel for reduction(+:plaq_tmp)
    for(L = 0; L < Dim; ++L){
        dec_to_bin(L, bin_state, dimstate);
        plaq_tmp+=bin_state[3]*norm(psi0[L]);
    }
    plaq[0]=plaq_tmp;

    // iterate solver
    #pragma omp parallel private(t,Lp,L,i,plaq_tmp)
    {
        vector<Complex> psi(psi0);
        vector<Complex> psi_aux(Dim,0.0);

        #pragma omp for 
        for(i = 1; i < n+1; ++i){
            t=i*dt;
            // apply phases
            for(Lp = 0; Lp < Dim; ++Lp)
                psi_aux[Lp]=exp(-iu*eigvals(Lp)*t)*psi0_mom[Lp];

            // transform back to standard basis
            for(L = 0; L < Dim; ++L){
                psi[L]=0.0;
                for(Lp = 0; Lp < Dim; ++Lp){
                    psi[L]+=psi_aux[Lp]*eigvecs(L,Lp);
                }
            }

            // compute plaquette on the state
            plaq_tmp=0.0;
            for(L = 0; L < Dim; ++L){
                dec_to_bin(L, bin_state, dimstate);
                plaq_tmp+=bin_state[3]*norm(psi[L]);
                //plaq[0]+= norm(psi[L]);
            }
            plaq[i] = plaq_tmp;
       }

    }

    tglob.toc();
    cout<<endl;
    cout<<"\b\b\b"<<DONE<<" in "<<tglob.toc()<<" seconds."<<endl;

    // save data
    FILE * ReTrPl_file = fopen((filestem+"_ReTrPl.dat").c_str(),"w");
    for(i = 0; i < n+1; ++i){
        fprintf(ReTrPl_file,"%.2f %.10lf\n",i*dt,plaq[i]);
    }
    fclose(ReTrPl_file);

//    cout<<"Evolution ..."<<flush;
//    tglob.tic();
//
//    //initialize the state with |0000000>
//    vector<Complex> psi0((int)pow(2,dimstate), 0.0);
//    psi0[0]={1.0,0.0};
//
//    vector<Complex> psi0_mom((int)pow(2,dimstate), 0.0);
//    
//    int i;
//    int Lp,L;
//    double t=0.0;
//    double numden_tmp;
//    vector<double> numden(n+1,0.0);
//
//    numden_tmp=0.0;
//    vector<int> bin_state(dimstate);
//    for(Lp=0;Lp<(int)pow(2,dimstate);++Lp){
//            for(L=0;L<(int)pow(2,dimstate);++L){
//                psi0_mom[Lp] += eigvecs(L,Lp)*psi0[L];
//            }
//    }
//
//    for(L=0;L<(int)pow(2,dimstate);++L){
//        dec_to_bin(L, bin_state, dimstate);
//        numden_tmp += bin_state[3]*norm(psi0[L]); 
//    }
//    numden[0]=numden_tmp;
//    numden_tmp=0.0;
//
//    vector<Complex> psi(psi0);
//    vector<Complex> psi_aux((int)pow(2,dimstate),0.0);
//
//    for(i=0;i<n+1;++i){
//        t=i*dt;
//        
//        for(Lp=0;Lp < Dim;++Lp){
//            psi_aux[Lp]=exp(-iu*eigvals(Lp)*t)*psi0_mom[Lp];
//        }
//
//        for(L=0;L<(int)pow(2,dimstate);++L){
//            psi[L]=0.0;
//            for(Lp=0;Lp<(int)pow(2,dimstate);++Lp){
//                psi[L]+=psi_aux[Lp]*eigvecs(L,Lp);
//            }
//        }
//        numden_tmp=0;
//        for(L=0;L<(int)pow(2,dimstate);++L){
//            dec_to_bin(L, bin_state, dimstate);
//            numden_tmp += bin_state[0]*norm(psi[L]);
//        }   
//        numden[i]=numden_tmp;
//    }
//
//    tglob.toc();
//    cout<<endl;
//    cout<<"\b\b\b"<<DONE<<" in "<<tglob.toc()<<" seconds."<<endl;
//
//
//    FILE * numdenfile = fopen((filestem+"_numDen.dat").c_str(),"w");
//    for(i = 0; i < n+1; ++i){
//        fprintf(numdenfile,"%.2f %.10lf\n",i*dt,numden[i]);
//    }
//    fclose(numdenfile);















    
    
    
    return 0;
}
