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
#include "io.hpp"
#include "suqa.cuh"



int main() {

	suqa::setup(5);

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


    suqa::clear();

	return 0;
}
