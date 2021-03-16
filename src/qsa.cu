#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <bits/stdc++.h>
#include <unistd.h>
#include <cmath>
#include <cassert>
#include "Rand.hpp"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "io.hpp"
#include "parser_qsa.hpp"
#include "suqa.cuh"
#include "system.cuh"
#include "qsa.cuh"

using namespace std;

// simulation parameters
double beta;

// defined in src/system.cu
void qsa_init_state();
void evolution_measure(const double& t, const int& n);
void evolution_szegedy(const double& t, const int& n);
void evolution_tracing(const double& t, const int& n);


arg_list args;

#ifdef GATECOUNT
GateCounter gctr_global("global");
GateCounter gctr_annstep("ann. step");
GateCounter gctr_annealing("annealing");
GateCounter gctr_sample("sample");
GateCounter gctr_measure("measure");
#endif

void save_measures(string outfilename, int rej){
	FILE * fil = fopen(outfilename.c_str(), "a");
	for(uint ei = 0; ei < qsa::E_measures.size(); ++ei){
		fprintf(fil, "%.16lg	%.16lg	%d\n", qsa::E_measures[ei],qsa::X_measures[ei],rej);
	}
	fclose(fil);
	qsa::E_measures.clear();
	qsa::X_measures.clear();
}

void cevolution_tracing(const double& t, const int& n, const uint& q_control, const bmReg& qstate){
    if(qstate.size()!=qsa::syst_qbits)
        throw std::runtime_error("ERROR: controlled evolution has wrong number of state qbits");

    suqa::activate_gc_mask({q_control});
    evolution_tracing(t, n);
    suqa::deactivate_gc_mask({q_control});
}


void apply_phase_estimation_tracing(const std::vector<uint>& q_state, const std::vector<uint>& q_target, const double& t, const uint& n){
    suqa::apply_h(q_target);
    // apply CUs
    for(int trg = q_target.size() - 1; trg > -1; --trg){
        double powr = (double)(1U << (q_target.size()-1-trg));

        cevolution_tracing(-powr*t, powr*n, q_target[trg], q_state);

        suqa::apply_u1(q_target[trg], -powr*t*qsa::t_PE_shift);
    }
    // apply QFT^{-1}
    qsa::qsa_qft_inverse(q_target);
}




int main(int argc, char** argv){

	if(argc < 6){
		printf("usage: %s <beta> <sampling> <num state qbits> <num ene qbits> <num szegedy qbits> <output file path>  [--seed <seed> (random)] [--annealing_sequences (100)]\n", argv[0]);
		exit(1);
	}

	parse_arguments(args, argc, argv);

	qsa::beta = args.beta;
	qsa::syst_qbits = (uint)args.syst_qbits;
	qsa::ene_qbits = (uint)args.ene_qbits;
	qsa::szegedy_qbits= (uint)args.szegedy_qbits;
	string outfilename(args.outfile);
	qsa::n_phase_estimation = args.pe_steps;
	qsa::iseed = args.seed;
//	uint lambda1_iterations= args.lambda1_iterations;
//	uint szegedy_iterations= args.szegedy_iterations;
	uint sampling= args.sampling;
	uint annealing_sequences= args.annealing_sequences;

	if(qsa::iseed>0)
		qsa::rangen.set_seed(qsa::iseed);

	qsa::iseed = qsa::rangen.get_seed();

	qsa::nqubits = qsa::syst_qbits + qsa::ene_qbits + qsa::szegedy_qbits + 1;
	qsa::Dim = (1U << qsa::nqubits);
	qsa::ene_levels = (1U << qsa::ene_qbits);
	double szegedy_levels = (1U << qsa::szegedy_qbits);
//	qsa::syst_levels = (1U << qsa::syst_qbits);

	qsa::t_PE_shift = args.ene_min;
	qsa::t_PE_factor = (qsa::ene_levels-1)/(double)(qsa::ene_levels*(args.ene_max-args.ene_min));//(qsa::ene_levels-1)
	qsa::t_phase_estimation = qsa::t_PE_factor*8.*atan(1.0); // 2*pi*t_PE_factor

    qsa::t_PE_shift_szegedy = args.ene_min_szegedy;
	qsa::t_PE_factor_szegedy = (szegedy_levels-1)/(double)((szegedy_levels)*(args.ene_max_szegedy-args.ene_min_szegedy));//(szegedy_levels-1)
	qsa::t_phase_estimation_szegedy = qsa::t_PE_factor_szegedy*8.*atan(1.0);


    suqa::print_banner();
	cout<<"arguments:\n"<<args<<endl;
	auto t_start = std::chrono::high_resolution_clock::now();
	// Initialization of utilities


	suqa::setup(qsa::nqubits);
	qsa::setup();
    DEBUG_CALL(cout<<"Init state: "<<endl);
    DEBUG_READ_STATE();

#ifdef GATECOUNT
    suqa::gatecounters.add_counter(&gctr_global);
    suqa::gatecounters.add_counter(&gctr_annstep);
    suqa::gatecounters.add_counter(&gctr_annealing);
    suqa::gatecounters.add_counter(&gctr_sample);
    suqa::gatecounters.add_counter(&gctr_measure);

    gctr_global.new_record();
#endif

	// Initialization:
	// known eigenstate of the system (see src/system.cu)
	//TODO: make it an args option?
	//   uint perc_mstep = (qsa::metro_steps+19)/20; // batched saves
	if( access( outfilename.c_str(), F_OK ) == -1 ){
		FILE * fil = fopen(outfilename.c_str(), "w");
		//        fprintf(fil, "# E%s\n",(qsa::Xmatstem!="")?" A":"");
		fprintf(fil, "# E A\n");
		fclose(fil);
	}


	int rejection=0;
	int rejection1=0;
	//	int rejection2=0;
    int itctr=0;
	double mean=0.0,std=0.0;
	//annealing_sequences=(int)(1000*qsa::beta/3.5);
	double beta_i=qsa::beta/((double)annealing_sequences);
	for( uint iiii=0; iiii< sampling; ){
#ifdef GATECOUNT
        gctr_sample.new_record();
        gctr_annealing.new_record();
#endif
		double Ene_measure;
		uint c_ac=0U;
		std::vector<uint> c_Ene_test(qsa::szegedy_qbits,0);
		std::vector<uint> c_Ene(qsa::szegedy_qbits,0);
		std::vector<uint> c_meas(qsa::ene_qbits,0);

		int control=0;

		while(control==0){
			int control2=0;
			uint move= qsa::draw_C();
			qsa::reset_non_syst_qbits();
			qsa_init_state();
            DEBUG_CALL(cout<<"Init syst state: "<<endl);
            DEBUG_READ_STATE();


			for( uint s=0U; s< annealing_sequences; s++){
#ifdef GATECOUNT
                gctr_annstep.new_record();
#endif
                DEBUG_CALL(cout<<"After annealing: "<<s<<endl);
                move= qsa::draw_C();
				qsa::apply_PE_szegedy(qsa::bm_states, qsa::bm_szegedy,(double)(s+1)*beta_i,move);
                DEBUG_CALL(cout<<"After PE szegedy: "<<endl);
                DEBUG_READ_STATE();

				suqa::measure_qbits(qsa::bm_szegedy, c_Ene_test, qsa::extract_rands(qsa::szegedy_qbits));
                DEBUG_CALL(cout<<"After measure: "<<endl);
                DEBUG_READ_STATE();

#ifdef GATECOUNT
        gctr_annstep.deactivate();
#endif

				if(qsa::creg_to_uint(c_Ene_test)!=0U  ) {
					control2=1;
					rejection1++;
					printf("break after %d steps   eigenvalue:%d   \n",s, qsa::creg_to_uint(c_Ene_test));

                    break;
				}
			}

			if (control2==0) control=1;
		}
        DEBUG_CALL(cout<<"CETS: "<<endl);
        DEBUG_READ_STATE();

#ifdef GATECOUNT
        gctr_annealing.deactivate();
#endif

//FINDING W EIGENVALUE
/*
qsa::reset_non_syst_qbits();
init_state(qsa::Dim);
qsa::apply_PE_szegedy(qsa::bm_states, qsa::bm_szegedy,(double)(1)*beta_i,qsa::draw_C());
suqa::measure_qbits(qsa::bm_szegedy, c_Ene_test, qsa::extract_rands(qsa::szegedy_qbits));
Ene_measure=qsa::creg_to_uint(c_Ene_test)/szegedy_levels;
qsa::E_measures.push_back(Ene_measure);
qsa::X_measures.push_back(measure_X(qsa::State,qsa::rangen));
cout<<"iteration: "<<iiii+1<<"/"<<sampling<<endl;
save_measures(outfilename,0);
*/
//DEBUG_READ_STATE(qsa::State);
    //  	suqa::measure_qbit(qsa::bm_acc, c_ac, qsa::rangen.doub());

		if(c_ac==0U){
//apply_phase_estimation_tracing(qsa::bm_states,qsa::bm_enes,  qsa::t_phase_estimation, qsa::n_phase_estimation);
//	suqa::measure_qbits(qsa::bm_enes, c_meas, qsa::extract_rands(qsa::ene_qbits));
#ifdef GATECOUNT
            gctr_measure.new_record();
#endif

			qsa::apply_phase_estimation_measure(qsa::bm_states,qsa::bm_szegedy, qsa::t_phase_estimation_szegedy, qsa::n_phase_estimation);
            DEBUG_CALL(cout<<"after apply_phase_estimation_measure()"<<endl);
            DEBUG_READ_STATE();
			suqa::measure_qbits(qsa::bm_szegedy, c_Ene, qsa::extract_rands(qsa::szegedy_qbits));
            DEBUG_CALL(cout<<"after measure_qbits()"<<endl);
            DEBUG_READ_STATE();

			Ene_measure= qsa::t_PE_shift_szegedy+qsa::creg_to_uint(c_Ene)/(double)(qsa::t_PE_factor_szegedy*szegedy_levels);

            printf("measured eigenvalue:%f\n",Ene_measure);
            itctr++;
            mean+= 	Ene_measure;
            std+= 	Ene_measure*Ene_measure;
            if(itctr>1)
                printf("mean=%f, std=%f; expected mean=%f\n", mean/itctr, sqrt((std-mean*mean/itctr)/((itctr)*(itctr-1))), (6-6*exp(4*qsa::beta))/(2+6*exp(4*qsa::beta)));
            qsa::E_measures.push_back(Ene_measure);
            qsa::X_measures.push_back(measure_X(qsa::rangen));
            DEBUG_CALL(cout<<"after measure_X()"<<endl);
            DEBUG_READ_STATE();

            cout<<"iteration: "<<iiii+1<<"/"<<sampling<<endl;
            save_measures(outfilename,rejection1);

#ifdef GATECOUNT
            gctr_measure.deactivate();
#endif
            ++iiii;
        }
        else rejection++;
	}
	printf("n of rejection step cause by acceptance qubit= %d\n", rejection);
	printf("n of rejection step cause by energy qubits= %d\n", rejection1);
		//printf("n of rejection step cause by not projection= %d\n", rejection2);
	printf("n of total call of W= %d\n", annealing_sequences*sampling);

	cout<<endl;
	qsa::clear();
	suqa::clear();
	cout<<"\nall fine :)\n"<<endl;

	cout<<"\n\tSuqa!\n"<<endl;
	auto t_end = std::chrono::high_resolution_clock::now();
	double secs_passed = (1./1000.)*std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cout<<"All [DONE] in "<<secs_passed<<" seconds"<<endl;
	return 0;

}
