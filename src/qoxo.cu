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

#ifdef GPU
#define NUM_THREADS 128
#define MAXBLOCKS 65535
uint suqa::threads;
uint suqa::blocks;
cudaStream_t suqa::stream1, suqa::stream2;
#endif

const int NQ = 19;
const int Dim = 1U << NQ;
const uint bm_win = 18;
const bmReg bm_slots = { 0U,1U,2U,3U,4U,5U,6U,7U,8U,9U,10U,11U,12U,13U,14U,15U,16U,17U };

pcg rangen;

struct Move {
    string type;
    int slot[2];

	void apply_move(int offset) {
		if (type.compare("flip") == 0) {
			// applies flip only if not flipped already by the other player
			suqa::apply_cx(slot[0]*2+(1-offset),slot[0] * 2+offset,0U);
		} else if(type.compare("mix")==0) { // mix type
			// applies mix only if not flipped already by the other player
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(slot[0] * 2+offset);
			suqa::apply_h(slot[1] * 2+offset);
            suqa::deactivate_gc_mask({bm_win});
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		} else if(type.compare("bell")==0) { // mix type
			// applies bell only if not flipped already by the other player
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(slot[0] * 2+offset);
			suqa::apply_cx(slot[0] * 2+offset, slot[1] * 2+offset);
            suqa::deactivate_gc_mask({bm_win});
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		}
		
	}
};

void get_player_move(Move& move) {
    bool good_format = false;
    while (!good_format) {
		cin >> move.type;
        if (cin.fail() || (move.type.compare("flip")!=0 && move.type.compare("mix")!=0 && move.type.compare("bell")!=0)) {
            cin.clear();

        } else if(move.type.compare("flip")==0){
            cin >> move.slot[0];
            if(cin.fail() || move.slot[0]<0 || move.slot[0]>8)
				cin.clear();
            else
				good_format = true;
        } else if(move.type.compare("mix")==0 || move.type.compare("bell")==0){
            cin >> move.slot[0] >> move.slot[1];
            if(cin.fail() || move.slot[0]<0 || move.slot[0]>8 || move.slot[1]<0 || move.slot[1]>8 || move.slot[0]==move.slot[1])
				cin.clear();
            else
				good_format = true;
        }
        if(!good_format)
            cout << "bad move format;\nmake move: "<<flush;
    }
}



const vector<vector<uint>> winsets = { {0U, 2U, 4U}, //rows
                                        {6U, 8U, 10U},
                                        {12U, 14U, 16U},
                                        {0U, 6U, 12U}, //columns
                                        {2U, 8U, 14U},
                                        {4U, 10U, 16U},
                                        {0U, 8U, 16U}, //diagonals
                                        {4U, 8U, 12U} };

//void print_classical_state(const vector<uint>& creg) {
//    for (uint r = 0U; r < 3U; ++r) {
//		for (uint c = 0U; c < 3U; ++c) {
//            printf("(%u, %u) ", creg[(c + r * 3U) * 2U], creg[(c + r * 3U) * 2U + 1U]);
//		}
//        printf("\n");
//    }
//	printf("\n");
//}

// measure on the win states
bool check_win(uint pl) {
    uint win_meas=0U;
//    for (uint offset = 0U; offset < 2U; ++offset) {
    uint offset=pl;
        for (const auto& triple : winsets) {
            suqa::apply_mcx({ triple[0] + offset, triple[1] + offset,triple[2] + offset }, bm_win);
			suqa::measure_qbit(bm_win, win_meas, rangen.doub());
            if (win_meas == 1U) {
//                offset = 2U;
                break;
            }
        }
//    }
//    DEBUG_CALL(printf("COLLAPSE\n\n"));
    if (win_meas == 1U) {
        printf(AYELLOW "COLLAPSE -> PLAYER %d WINS\n\n" ARESET,pl+1);
        DEBUG_READ_STATE(suqa::state);
    }

//    if (win_meas == 1U) {
//        suqa::apply_x(state, bm_win);
//        vector<uint> c_reg(18);
//        vector<double> rgens(18);
//        for (auto& el : rgens) el = rangen.doub();
//		suqa::measure_qbits(state, bm_slots, c_reg, rgens);
////        printf("Win state:\n");
////        print_classical_state(c_reg);
//        DEBUG_READ_STATE(state);
//    }

    return win_meas==1U;
}

// quantum turns for both players
bool double_turn(int pl) {
    static int turn_counter = 1;
    cout << "turn " << (turn_counter++) << endl;
    
    Move move;
    cout << "\nPlayer "<<pl+1<<", make move: " << flush;
    get_player_move(move);
    move.apply_move(pl);
    printf("\n");
//	DEBUG_CALL(printf("game state:\n"));
    bool win = check_win(pl);

    if(!win){
        DEBUG_READ_STATE(suqa::state);
    }

    return win;

//    cout << "\nPlayer 2, make move: " << flush;
//    get_player_move(move);
//    
//    move.apply_move(1);
//	DEBUG_CALL(printf("game state:\n"));
//	DEBUG_READ_STATE(state);
}

void game() {
//    suqa::init_state();
	DEBUG_READ_STATE(suqa::state);


    bool win = false;
    int pl = 0;
    while (!win) {
        win  = double_turn(pl);
        pl=1-pl;
    }

}


int main(int argc, char** argv) {
    (void)argc,(void)argv;
    
    printf("Welcome to QOXO\n");
    printf("a quantum generalization of the game OXO (aka tic-tac-toe)\n");
    printf("\n0|1|2\n");
    printf("_____\n");
    printf("3|4|5\n");
    printf("______\n");
    printf("6|7|8\n\n");
    printf("Notation:\np: generic player, a: antagonist player;\n");
//    printf("| |_{i} :empty slot at site i; |p|_i : i site occupied by p's symbol\n");
//    printf("p) <movename> <i> [<j>] : move named <movename> made by player p and at the site <i> and possibly also <j> (depending on the move).\n");
    printf("\n\nRules: each player can perform certain types of moves, called 'flip', 'mix' or 'bell'.\n");
    printf("p) flip <i> : \\bar{C}^{(a)}_i X^{(p)}_i [like the usual classical move in OXO]\n");
//    printf("p) mix <i> <j> : | , >, |a| -> |a| [like the usual classical move in OXO]\n");
    printf("p) mix <i> <j> : \\bar{C}^{(a)}_{i,j}-(H^{(p)}_i H^{(p)}_j)\n");
    printf("p) bell <i> <j> : \\bar{C}^{(a)}_{i,j}-(H^{(p)}_i C^{(p)}_i-X^{(p)}_j)\n\n");

//    if (argc < 1) {
//        printf("usage: %s <g_beta> <total_steps> <trotter_stepsize> <outfile>\n", argv[0]);
//        exit(1);
//    }


#ifdef GPU
    suqa::threads = NUM_THREADS;
    suqa::blocks = (Dim + suqa::threads - 1) / suqa::threads;
    if (suqa::blocks > MAXBLOCKS) suqa::blocks = MAXBLOCKS;
    printf("blocks: %u, threads: %u\n", suqa::blocks, suqa::threads);
#endif

    suqa::setup(NQ);

    rangen.set_seed(time(NULL));
    rangen.randint(0, 3);

    game();
    
//    FILE* outfile;


//        suqa::prob_filter(state, bm_spin, { 1U,1U,1U }, p111);

    suqa::clear();

    return 0;
}
