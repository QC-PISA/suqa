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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "io.hpp"
#include "suqa.cuh"
//#include "system.cuh"
#include "Rand.hpp"


using namespace std;

#define NUM_THREADS 128
#define MAXBLOCKS 65535
uint suqa::threads;
uint suqa::blocks;
cudaStream_t suqa::stream1, suqa::stream2;

const int NQ = 19;
const int Dim = 1U << NQ;
ComplexVec state;
const uint bm_win = 18;
const bmReg bm_slots = { 0U,1U,2U,3U,4U,5U,6U,7U,8U,9U,10U,11U,12U,13U,14U,15U,16U,17U };

pcg rangen;

void deallocate_state(ComplexVec& state) {
    if (state.data != nullptr) {
        HANDLE_CUDACALL(cudaFree(state.data));
    }
    state.vecsize = 0U;
}

void allocate_state(ComplexVec& state, uint Dim) {
    if (state.data != nullptr or Dim != state.vecsize)
        deallocate_state(state);


    state.vecsize = Dim;
    HANDLE_CUDACALL(cudaMalloc((void**)&(state.data), 2 * state.vecsize * sizeof(double)));
    // allocate both using re as offset, and im as access pointer.
    state.data_re = state.data;
    state.data_im = state.data_re + state.vecsize;
}

__global__ void initialize_state(double *state_re, double *state_im, uint len){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x*blockDim.x;
    }
    if(blockIdx.x*blockDim.x+threadIdx.x==1){
        state_re[0] = 1.0;
        state_im[0] = 0.0;
    }
}

struct Move {
    string type;
    int slot[2];

	void apply_move(int offset) {
		if (type.compare("flip") == 0) {
			// applies flip only if not flipped already by the other player
			suqa::apply_cx(state, slot[0]*2+(1-offset),slot[0] * 2+offset,0U);
		} else { // mix type
			// applies mix only if not flipped already by the other player
            suqa::apply_mcx(state, { static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(state, slot[0] * 2+offset);
			suqa::apply_h(state, slot[1] * 2+offset);
//			suqa::apply_cx(state, slot[0] * 2+offset, slot[1] * 2+offset);
            suqa::deactivate_gc_mask();
            suqa::apply_mcx(state, { static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		}
		
	}
};

void get_player_move(Move& move) {
    bool good_format = false;
    while (!good_format) {
		cin >> move.type;
        if (cin.fail() || (move.type.compare("flip")!=0 && move.type.compare("mix")!=0)) {
            cin.clear();

            cout << "bad move format;\nmake move: "<<flush;
        } else if(move.type.compare("flip")==0){
            cout << "flip selected" << endl;
            cin >> move.slot[0];
            if(cin.fail() || move.slot[0]<0 || move.slot[0]>8)
				cin.clear();
            else
				good_format = true;
        } else if(move.type.compare("mix")==0){
            cout << "mix selected" << endl;
            cin >> move.slot[0] >> move.slot[1];
            if(cin.fail() || move.slot[0]<0 || move.slot[0]>8 || move.slot[1]<0 || move.slot[1]>8 || move.slot[0]==move.slot[1])
				cin.clear();
            else
				good_format = true;
        }
    }
}


// quantum turns for both players
void double_turn() {
    static int turn_counter = 1;
    cout << "turn " << (turn_counter++) << endl;
    
    Move move;
    cout << "\nPlayer 1, make move: " << flush;
    get_player_move(move);
    move.apply_move(0);

    cout << "\nPlayer 2, make move: " << flush;
    get_player_move(move);
    move.apply_move(1);
}

const vector<vector<uint>> winsets = { {0U, 2U, 4U}, //rows
                                        {6U, 8U, 10U},
                                        {12U, 14U, 16U},
                                        {0U, 6U, 12U}, //columns
                                        {2U, 8U, 14U},
                                        {4U, 10U, 16U},
                                        {0U, 8U, 16U}, //diagonals
                                        {4U, 8U, 12U} };

void print_classical_state(const vector<uint>& creg) {
    for (uint r = 0U; r < 3U; ++r) {
		for (uint c = 0U; c < 3U; ++c) {
            printf("(%u, %u) ", creg[(c + r * 3U) * 2U], creg[(c + r * 3U) * 2U + 1U]);
		}
        printf("\n");
    }
	printf("\n");
}

// measure on the win states
bool check_win() {
    // rows
    for (uint offset = 0U; offset < 2U; ++offset) {
        for (const auto& triple : winsets) {
            suqa::apply_mcx(state, { triple[0] + offset, triple[1] + offset,triple[2] + offset }, bm_win);
        }
    }

    uint win_meas;
    suqa::measure_qbit(state, bm_win, win_meas, rangen.doub());
    if (win_meas == 1U) {
        suqa::apply_x(state, bm_win);
        vector<uint> c_reg(18);
        vector<double> rgens(18);
        for (auto& el : rgens) el = rangen.doub();
		suqa::measure_qbits(state, bm_slots, c_reg, rgens);
        printf("Win state:\n");
        print_classical_state(c_reg);
    }

    return win_meas==1U;
}

void game() {
    initialize_state<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im,Dim);

    bool win = false;
    while (!win) {
        double_turn();
        win=check_win();
		DEBUG_CALL(printf("game state:\n"));
		DEBUG_READ_STATE(state);
    }

}


int main(int argc, char** argv) {
    
    printf("Welcome to QOXO\n");

//    if (argc < 1) {
//        printf("usage: %s <g_beta> <total_steps> <trotter_stepsize> <outfile>\n", argv[0]);
//        exit(1);
//    }

    suqa::threads = NUM_THREADS;
    suqa::blocks = (Dim + suqa::threads - 1) / suqa::threads;
    if (suqa::blocks > MAXBLOCKS) suqa::blocks = MAXBLOCKS;
    printf("blocks: %u, threads: %u\n", suqa::blocks, suqa::threads);


    allocate_state(state, Dim);

    rangen.set_seed(time(NULL));
    rangen.randint(0, 3);

    suqa::setup(Dim);

    game();
    
//    FILE* outfile;

    DEBUG_CALL(printf("initial state:\n"));
    DEBUG_READ_STATE(state);

//        suqa::prob_filter(state, bm_spin, { 1U,1U,1U }, p111);

    suqa::clear();

    deallocate_state(state);


    return 0;
}
