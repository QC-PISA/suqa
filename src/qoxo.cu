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
#include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
#include <unistd.h> // for STDOUT_FILENO

#define ARED     "\x1b[31m"
#define AGREEN   "\x1b[32m"
#define AYELLOW  "\x1b[33m"
#define ABLUE    "\x1b[34m"
#define AMAGENTA "\x1b[35m"
#define ACYAN    "\x1b[36m"
#define ARESET   "\x1b[0m"

const char qoxocharmap[3] = {' ','O','X'};

#ifdef SPARSE
void qoxo_print(double *v_re, double *v_im, std::vector<uint> actives){
#else
void qoxo_print(double *v_re, double *v_im, uint vecsize){
#endif


    struct winsize wsize;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsize);

    /* size.ws_row is the number of rows, size.ws_col is the number of columns. */


    // for non-contiguous even-odd entries corresponding to real and imag parts
    using  uintdoub = std::pair<uint,double>;
    std::vector<uintdoub> state_norm;
#ifdef SPARSE
    for(const uint& i : actives){
#else
    for(uint i=0; i<vecsize; ++i){
#endif
       std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-8){
           state_norm.push_back(std::make_pair(i,norm(var)));
       }
    }
    std::sort(state_norm.begin(),state_norm.end(),[](const uintdoub &a, const uintdoub &b){return a.second > b.second;});

    std::vector<std::vector<char>> gms(state_norm.size(),std::vector<char>(9));    
    for(size_t idx = 0; idx< state_norm.size(); ++idx){
        auto uid = state_norm[idx];
        uint game_stidx = uid.first;
        for(uint i=0U; i<9U; ++i){
            gms[idx][i] = qoxocharmap[(game_stidx & 3U)]; // 0 -> empty, 1-> first pl, 2-> second pl, 3-> inactive
            game_stidx>>=2U;
        }
    }

    printf("\n");
    size_t max_ngxrow = wsize.ws_col/12;
    size_t max_ngxcol = (state_norm.size()+max_ngxrow-1)/max_ngxrow;
    for(size_t Row = 0; Row<max_ngxcol;++Row){
        size_t ngxrow = (Row==max_ngxcol-1)?(state_norm.size()%max_ngxrow):max_ngxrow;

        for(size_t Col=0; Col<ngxrow; ++Col) printf("------------");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][0],gms[idx][1],gms[idx][2]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("|   _____  |");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][3],gms[idx][4],gms[idx][5]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("|   _____  |");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][6],gms[idx][7],gms[idx][8]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|  " ACYAN "%5.1f%%" ARESET "  |",state_norm[idx].second*100);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("------------");
        printf("\n");
        
    }
    std::cout<<std::endl;
}

#ifdef GPU
#if !defined(NDEBUG) 
extern double *host_state_re, *host_state_im;
#define DEBUG_READ_STATE_QOXO() {\
    cudaMemcpyAsync(host_state_re,state.data_re,suqa::state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream1); \
    cudaMemcpyAsync(host_state_im,state.data_im,suqa::state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream2); \
    cudaDeviceSynchronize(); \
    qoxo_print((double*)host_state_re,(double*)host_state_im, suqa::state.size()); \
} 
#else  // NDEBUG
#define DEBUG_READ_STATE_QOXO()
#endif

#else // CPU
#ifdef SPARSE
#define DEBUG_READ_STATE_QOXO() {\
    qoxo_print((double*)suqa::state.data_re,(double*)suqa::state.data_im,suqa::actives); \
}
//    printf("vnorm = %.12lg\n",suqa::vnorm());
#else // NON SPARSE
#define DEBUG_READ_STATE_QOXO() {\
    qoxo_print((double*)suqa::state.data_re,(double*)suqa::state.data_im, suqa::state.size()); \
}
#endif // end SPARSE
#endif // end GPU


using namespace std;

void print_rules(){
    printf("Welcome to QOXO\n");
    printf("a quantum generalization of the game OXO (aka tic-tac-toe)\n");
    printf("\n1|2|3\n");
    printf("_____\n");
    printf("4|5|6\n");
    printf("______\n");
    printf("7|8|9\n\n");
    printf("Notation:\np: generic player, a: antagonist player;\n");
//    printf("| |_{i} :empty slot at site i; |p|_i : i site occupied by p's symbol\n");
//    printf("p) <movename> <i> [<j>] : move named <movename> made by player p and at the site <i> and possibly also <j> (depending on the move).\n");
    printf("\n\nRules: each player can perform certain types of moves, called 'flip', 'bell' and 'mix'.\n");
    printf("p) flip <i> : \\bar{C}^{(a)}_i X^{(p)}_i [like the usual classical move in OXO]\n");
    printf("p) split <i> <j>: \\bar{C}^{(a)}_i\\_(X^{(p)}_j H^{(p)}_i C^{(p)}_i\\_X^{(p)}_j)\n");
//    printf("p) mix <i> <j> : | , >, |a| -> |a| [like the usual classical move in OXO]\n");
    printf("p) bell <i> <j> : \\bar{C}^{(a)}_{i,j}\\_(H^{(p)}_i C^{(p)}_i\\_X^{(p)}_j)\n");
    printf("p) mix <i> <j> : \\bar{C}^{(a)}_{i,j}-(H^{(p)}_i H^{(p)}_j)\n\n");
}


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
		} else if(type.compare("split")==0) { // split type
			// applies bell only if not flipped already by the other player
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(slot[0] * 2+offset);
			suqa::apply_cx(slot[0] * 2+offset, slot[1] * 2+offset);
			suqa::apply_x(slot[1] * 2+offset);
            suqa::deactivate_gc_mask({bm_win});
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		} else if(type.compare("bell")==0) { // bell type
			// applies bell only if not flipped already by the other player
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(slot[0] * 2+offset);
			suqa::apply_cx(slot[0] * 2+offset, slot[1] * 2+offset);
            suqa::deactivate_gc_mask({bm_win});
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		} else if(type.compare("mix")==0) { // mix type
			// applies mix only if not flipped already by the other player
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
            suqa::activate_gc_mask({bm_win});
			suqa::apply_h(slot[0] * 2+offset);
			suqa::apply_h(slot[1] * 2+offset);
            suqa::deactivate_gc_mask({bm_win});
            suqa::apply_mcx({static_cast<uint>(slot[0] * 2 + (1 - offset)),static_cast<uint>(slot[1] * 2 + (1 - offset)) }, { 0U,0U }, bm_win);
		}
		
	}
};

void get_player_move(Move& move) {
    bool good_format = false;
    while (!good_format) {
		cin >> move.type;
        if (cin.fail() || (move.type.compare("flip")!=0 && move.type.compare("split")!=0 && move.type.compare("mix")!=0 && move.type.compare("bell")!=0)) {
            cin.clear();

        } else if(move.type.compare("flip")==0){
            cin >> move.slot[0];
            move.slot[0]--;
            if(cin.fail() || move.slot[0]<0 || move.slot[0]>8)
				cin.clear();
            else
				good_format = true;
        } else if(move.type.compare("split")==0 || move.type.compare("bell")==0 || move.type.compare("mix")==0){
            cin >> move.slot[0] >> move.slot[1];
            move.slot[0]--; move.slot[1]--;
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
        printf(AYELLOW "COLLAPSE -> PLAYER %c WINS\n\n" ARESET,qoxocharmap[pl+1]);
        DEBUG_READ_STATE();
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
    cout << "\nPlayer "<<qoxocharmap[pl+1]<<", make move: " << flush;
    get_player_move(move);
    move.apply_move(pl);
    printf("\n");
//	DEBUG_CALL(printf("game state:\n"));
    bool win = check_win(pl);

    if(!win){
        DEBUG_READ_STATE_QOXO();
    }

    return win;

//    cout << "\nPlayer 2, make move: " << flush;
//    get_player_move(move);
//    
//    move.apply_move(1);
//	DEBUG_CALL(printf("game state:\n"));
//	DEBUG_READ_STATE();
}

void game() {
//    suqa::init_state();
	DEBUG_READ_STATE_QOXO();


    bool win = false;
    int pl = 0;
    while (!win) {
        win  = double_turn(pl);
        pl=1-pl;
    }

}



int main(int argc, char** argv) {
    (void)argc,(void)argv;
    
    print_rules();


    suqa::setup(NQ);

    rangen.set_seed(time(NULL));
    rangen.randint(0, 3);

    game();
    
    suqa::clear();

    return 0;
}
