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

	ComplexVec state;
	suqa::setup(state,32);
	suqa::all_zeros(state);
	DEBUG_CALL(printf("Initial state:\n"));
	DEBUG_READ_STATE(state);

	suqa::clear(state);

	return 0;
}
