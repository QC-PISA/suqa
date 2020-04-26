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
//#include <bits/stdc++.h>
//#include <unistd.h>
#include <cmath>
#include <cassert>
#include <chrono>
#include "io.hpp"
#include "suqa.cuh"



int main(int argc, char** argv) {

	suqa::setup(5);
	suqa::init_state();


	DEBUG_CALL(printf("Initial state:\n"));
	DEBUG_READ_STATE(suqa::state);

	suqa::apply_x(3);
	DEBUG_CALL(printf("After apply_x(3):\n"));
	DEBUG_READ_STATE(suqa::state);

	suqa::apply_h(1);
	DEBUG_CALL(printf("After apply_h(1):\n"));
	DEBUG_READ_STATE(suqa::state);

	suqa::apply_cx(1,0);
	DEBUG_CALL(printf("After apply_cx(1,0):\n"));
	DEBUG_READ_STATE(suqa::state);

	suqa::clear();

	return 0;
}
