#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
//#include <bits/stdc++.h>
//#include <unistd.h>
#include <cmath>
#include <cassert>
#include <chrono>
#include "io.hpp"
#include "suqa.cuh"
#include "system.cuh"
#include "Rand.hpp"


using namespace std;

void self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3);
void inverse_self_plaquette(const bmReg& qr0, const bmReg& qr1, const bmReg& qr2, const bmReg& qr3);

int main(int argc, char** argv){
    if(argc<5){
       printf("usage: %s <g_beta> <total_steps> <trotter_stepsize> <outfile> [--init <initfile>]\n",argv[0]); 
       exit(0);
    }
    g_beta = stod(argv[1]); // (extern) def in src/system.cu
    int total_steps = atoi(argv[2]);
    double trotter_stepsize = stod(argv[3]);
    string outfilename = argv[4];
    bool with_initialization=argc>5;
    string initfile=with_initialization ? argv[6] : "";


	suqa::setup(syst_qbits);

    printf("arguments:\n g_beta = %.16lg\n total_steps = %d\n trotter_stepsize = %.16lg\n outfile = %s\n", g_beta, total_steps, trotter_stepsize, outfilename.c_str());


    // init
    pcg rangen;
//    rangen.set_seed(12345); //time(NULL));

    FILE * outfile;

    size_t dim=1<<syst_qbits;
    vector<double> re_coeff(dim);
    vector<double> im_coeff(dim);
    if(with_initialization){
        FILE * fl_init = fopen(initfile.c_str(),"r");
        for(size_t ii=0; ii<dim; ++ii){
            int ectrl=fscanf(fl_init,"%lg %lg\n",&re_coeff[ii],&im_coeff[ii]); (void)ectrl;
        }

        fclose(fl_init);
//        suqa::init_state(re_coeff,im_coeff);    
//    }else{
//        init_state();
    }
    if(with_initialization){
        suqa::init_state(re_coeff,im_coeff);    
    }else{
        init_state();
    }

    DEBUG_CALL(printf("initial state:\n"));
    DEBUG_READ_STATE();

    for(uint ii=0; ii<=(uint)total_steps; ++ii){
        double t = ii*trotter_stepsize;
        DEBUG_CALL(printf("time %.16lg\n", t));
        double plaq_val=0.0;
        double plaq_val_std=0.0;

        // legit way to perform measures
//        for(uint hit=0; hit<(uint)num_hits; ++hit){
//            printf("\thit %u\n", hit);
//            init_state(state, Dim, g_beta);
//            evolution(state, t, ii);
//            self_plaquette(state, bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
//            std::vector<uint> c(3);
//            suqa::measure_qbit(state,bm_qlink1[0],c[0],rangen.doub());
//            suqa::measure_qbit(state,bm_qlink1[1],c[1],rangen.doub());
//            suqa::measure_qbit(state,bm_qlink1[2],c[2],rangen.doub());
//            uint plaq_idx = 4*c[2]+2*c[1]+c[0];
//            double plaq_tmp = (plaq_idx==0)? 2.0 : ((plaq_idx==2)? -2.0 : 0.0);
//            plaq_val += plaq_tmp;
//            plaq_val_std += plaq_tmp*plaq_tmp;
//        }
//        plaq_val /=(double)num_hits;
//        plaq_val_std = sqrt((plaq_val_std/(double)num_hits - plaq_val*plaq_val)/(double)(num_hits-1));
//        fprintf(outfile, "%.16lg %d %.16lg %.16lg\n", t, num_hits, plaq_val, plaq_val_std);

//		suqa::apply_h(bm_spin[rangen.randint(0,3)]);
//        printf("random number= %d\n", rangen.randint(0,3));
//        DEBUG_READ_STATE();
        self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);

        //suqa::apply_h(state,  bm_spin[rangen.randint(0,3)]);
	
        double p000, p010;
        suqa::prob_filter(bm_qlink1, {0U,0U,0U}, p000);
        suqa::prob_filter(bm_qlink1, {0U,1U,0U}, p010);
        inverse_self_plaquette(bm_qlink1, bm_qlink0, bm_qlink2, bm_qlink0);
        printf("p000 = %.12lg; p010 = %.12lg\n", p000, p010);
        plaq_val = 2.0*(p000-p010);
        plaq_val_std = sqrt(4.0*(p000+p010)-plaq_val*plaq_val);
        outfile = fopen(outfilename.c_str(), "a");
        fprintf(outfile, "%.12lg %.12lg %.12lg\n", t, plaq_val, plaq_val_std);
        printf("%.12lg %.12lg %.12lg\n", t, plaq_val, plaq_val_std);
        fclose(outfile);

        if(ii<(uint)total_steps){
            evolution(trotter_stepsize, 1);
            DEBUG_CALL(printf("after evolution by t=%lg:\n",t));
            DEBUG_READ_STATE();
        }

        double discrepancy=0.0;
        for(size_t ii=0; ii<dim; ++ii){
            double initial_coeff=re_coeff[ii]*re_coeff[ii]+im_coeff[ii]*im_coeff[ii];
            double evo_coeff=suqa::state.data[ii]*suqa::state.data[ii]+suqa::state.data[dim+ii]*suqa::state.data[dim+ii];
            discrepancy+=abs(initial_coeff-evo_coeff);
        }
        std::cout<<"discrepancy with initial vector: "<<discrepancy<<std::endl;
    }
    
    suqa::clear();

    return 0;
}
