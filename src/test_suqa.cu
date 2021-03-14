#ifdef GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include <cassert>
#include <chrono>
#include "Rand.hpp"
#include "io.hpp"
#include "suqa.cuh"



int main() {

	suqa::setup(5);

#ifdef GATECOUNT
    GateCounter gctr_global("global");
    GateCounter gctr_region1("region1");
    GateCounter gctr_region2("region2");
    suqa::gatecounters.add_counter(&gctr_global);
    suqa::gatecounters.add_counter(&gctr_region1);
    suqa::gatecounters.add_counter(&gctr_region2);
#endif

//	suqa::init_state();
	DEBUG_CALL(printf("Initial state:\n"));
	DEBUG_READ_STATE();

	suqa::apply_x(3);
	DEBUG_CALL(printf("After apply_x(3):\n"));
	DEBUG_READ_STATE();

	suqa::apply_x(3);
	DEBUG_CALL(printf("After apply_x(3):\n"));
	DEBUG_READ_STATE();

	suqa::apply_h(1);
	DEBUG_CALL(printf("After apply_h(1):\n"));
	DEBUG_READ_STATE();

	suqa::apply_x(3);
	DEBUG_CALL(printf("After apply_x(3):\n"));
	DEBUG_READ_STATE();

	suqa::apply_h(1);
	DEBUG_CALL(printf("After apply_h(1):\n"));
	DEBUG_READ_STATE();

	suqa::apply_cx(3,0);
	DEBUG_CALL(printf("After apply_cx(3,0):\n"));
	DEBUG_READ_STATE();

	suqa::apply_cx(3,0);
	DEBUG_CALL(printf("After apply_cx(3,0):\n"));
	DEBUG_READ_STATE();

	suqa::apply_h(0);
	DEBUG_CALL(printf("After apply_h(0):\n"));
	DEBUG_READ_STATE();

	suqa::apply_h(1);
	DEBUG_CALL(printf("After apply_h(1):\n"));
	DEBUG_READ_STATE();

	suqa::apply_h({ 2, 4 });
	DEBUG_CALL(printf("After apply_h({2,4}):\n"));
	DEBUG_READ_STATE();

	suqa::apply_cx(0,4);
	DEBUG_CALL(printf("After apply_cx(0,4):\n"));
	DEBUG_READ_STATE();

	suqa::apply_u1(3,M_PI/3.0);
	DEBUG_CALL(printf("After apply_u1(3,M_PI/3.0):\n"));
	DEBUG_READ_STATE();

	suqa::clear();
	suqa::setup(1);
	DEBUG_CALL(printf("Clear and setup(1):\n"));
	DEBUG_READ_STATE();

	suqa::apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
	suqa::apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
	suqa::apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
	suqa::apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0},{PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();

	suqa::clear();
	suqa::setup(2);
	DEBUG_CALL(printf("Clear and setup(2):\n"));
	DEBUG_READ_STATE();
    suqa::apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
    suqa::apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
    suqa::apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();
    suqa::apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0);
	DEBUG_CALL(printf("After apply_pauli_TP_rotation({0,1},{PAULI_X,PAULI_X},M_PI/4.0):\n"));
	DEBUG_READ_STATE();

#ifdef GATECOUNT
    printf("\n\nTest gatecounters:\n\n");

	suqa::clear();
	suqa::setup(5);

    pcg rangen;

    gctr_global.new_record(); // activates the counter and set a new record
    for(uint i=0; i<10;){
        gctr_region1.new_record();

        // select two random qubits, two pauli matrices and a random phase

        uint c;
        for(uint j=0; j<3; ++j){
            gctr_region2.new_record();

            uint q1=rangen.randint(0,5);        
            uint q2=(q1+rangen.randint(1,5))%5;
            uint p1=rangen.randint(1,4);
            uint p2=rangen.randint(1,4);
            double theta = rangen.doub()*M_PI;
            
            suqa::apply_pauli_TP_rotation({q1,q2},{p1,p2},theta);
            printf("apply_pauli_TP_rotation({%u,%u},{%u,%u},%.3lf)\n",q1,q2,p1,p2,theta);
            suqa::measure_qbit(0,c,rangen.doub());
            printf("measure on 0: %u\n",c);
            if(c==0){
                break;
            }
        }
        gctr_region2.deactivate();
        if(c==1){
            suqa::apply_h(0);
            printf("apply_h(0):\n");
            ++i;
        }
    }
    printf("\n\nGate counts\n\n");
    suqa::gatecounters.print_counts();

    printf("\n\nGate averages\n\n");
    suqa::gatecounters.print_averages();
#endif

    suqa::clear();

	return 0;
}
