#include <algorithm>
#include <random>
#include <iostream>
#include <map>
#include <unordered_map>
#include <climits>
#include <cassert>
#include <cstring>
#include <string>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include "board.hpp"
#include "engine.hpp"
// #include<fstream>
#include<chrono>
#define curtime = std::chrono::system_clock::now().time_since_epoch().count()
using namespace std;
using feature = double (*)(Board& b, bool for_white); 
const double defaultalphaval = -1e18; const double defaultbetaval = 1e18;
const int QUAISCENCE_DEPTH = 3;
void humanplay(const Board& b, Engine *e);
PlayerColor our_color;
pair<U16,double> minimax(Board &b, Engine &e, int depth); vector<int> temp;
pair<U16,double> alpha_beta_minimax_old(Board &b, Engine &e, int depth,vector<U16> &path ,double alpha = defaultalphaval, double beta = defaultbetaval);
pair<U16,double> alpha_beta_minimax(Board &b, Engine &e, int cur_depth,vector<U16> &path,const vector<U16> &prev_best_moves,bool on_prev_best_path,double alpha = defaultalphaval, double beta = defaultbetaval); // void get_move_sequence(vector<U16>& moveseq,Board& b, Engine& e, int depth);
pair<U16,double> queiscence_alpha_beta_minimax(Board &b, Engine &e, int cur_depth,vector<U16> &path,double alpha = defaultalphaval,double beta = defaultbetaval);
double evaluate(Board& b,  std::unordered_map<string,feature>& features,std::unordered_map<string,double> &weight_map);
void update_weights(Board &b, double actual_cost, int at_depth, U8 move);
void get_weights_from_file(string filename = "weights/w"); 
void write_weights_to_file(string filename = "weights/w");



int qmode = 0; int debugmode = 1; int should_update_weights = 0; double learning_rate = 0.02; 
const int checkmate_score = 1e8; 
vector<int> stats_alpha_beta = {0,0,0,0,0,0}; //for all depths. 
//constructor that runs initially.
Engine::Engine()
{
    this->should_update = should_update_weights; //taking the value from the global variable.
    get_weights_from_file(); 
}
Engine::Engine(int should_train)
{
    this->should_update = should_train;
    get_weights_from_file();
}
// int depth = 4; //instead of setting a fixed depth, we can increase it as the game progresses.
int min_depth = 1; //sets up a minimum depht for us to look at, and will do iterative deepening after b point.
double time_per_move = 50; //sets up the time per move, need to finish within b time.
int max_depth = 50;
int move_number = 0;
//removed const from feature. Gotta be more careful.
#pragma region private_code

namespace private_code{
constexpr U8 cw_90[64] = {
    48, 40, 32, 24, 16, 8,  0,  7,
    49, 41, 33, 25, 17, 9,  1,  15,
    50, 42, 18, 19, 20, 10, 2,  23,
    51, 43, 26, 27, 28, 11, 3,  31,
    52, 44, 34, 35, 36, 12, 4,  39,
    53, 45, 37, 29, 21, 13, 5,  47,
    54, 46, 38, 30, 22, 14, 6,  55,
    56, 57, 58, 59, 60, 61, 62, 63
};

constexpr U8 acw_90[64] = {
     6, 14, 22, 30, 38, 46, 54, 7,
     5, 13, 21, 29, 37, 45, 53, 15,
     4, 12, 18, 19, 20, 44, 52, 23,
     3, 11, 26, 27, 28, 43, 51, 31,
     2, 10, 34, 35, 36, 42, 50, 39,
     1,  9, 17, 25, 33, 41, 49, 47,
     0,  8, 16, 24, 32, 40, 48, 55,
    56, 57, 58, 59, 60, 61, 62, 63
};

constexpr U8 cw_180[64] = {
    54, 53, 52, 51, 50, 49, 48, 7,
    46, 45, 44, 43, 42, 41, 40, 15,
    38, 37, 18, 19, 20, 33, 32, 23,
    30, 29, 26, 27, 28, 25, 24, 31,
    22, 21, 34, 35, 36, 17, 16, 39,
    14, 13, 12, 11, 10,  9,  8, 47,
     6,  5,  4,  3,  2,  1,  0, 55,
    56, 57, 58, 59, 60, 61, 62, 63
};

constexpr U8 id[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63
};

#define cw_90_pos(p) cw_90[p]
#define cw_180_pos(p) cw_180[p]
#define acw_90_pos(p) acw_90[p]
#define cw_90_move(m) move_promo(cw_90[getp0(m)], cw_90[getp1(m)], getpromo(m))
#define acw_90_move(m) move_promo(acw_90[getp0(m)], acw_90[getp1(m)], getpromo(m))
#define cw_180_move(p) move_promo(cw_180[getp0(m)], cw_180[getp1(m)], getpromo(m))
#define color(p) ((PlayerColor)(p & (WHITE | BLACK)))

std::unordered_set<U16> transform_moves(const std::unordered_set<U16>& moves, const U8 *transform) {

    std::unordered_set<U16> rot_moves;

    for (U16 move : moves) {
        rot_moves.insert(move_promo(transform[getp0(move)], transform[getp1(move)], getpromo(move)));
    }

    return rot_moves;
}

std::unordered_set<U16> construct_bottom_rook_moves_with_board(const U8 p0, const U8* board) {

    int left_rook_reflect[7] = {0, 8, 16, 24, 32, 40, 48};
    PlayerColor color = color(board[p0]);
    std::unordered_set<U16> rook_moves;
    bool refl_blocked = false;

    if (p0 < 8 || p0 == 13) {
        if (!(board[p0+pos(0,1)] & color)) rook_moves.insert(move(p0, p0+pos(0,1))); // top
        if (p0 == 1) { // top continued on the edge
            for (int y = 1; y<=6; y++) {
                U8 p1 = pos(1, y);
                if (board[p1]) {
                    if (board[p1] & color) break;         // our piece
                    else rook_moves.insert(move(p0, p1)); // their piece - capture
                    break;
                }
                else rook_moves.insert(move(p0, p1));
            }
        }
    }
    else {
        if (!(board[p0-pos(0,1)] & color)) rook_moves.insert(move(p0, p0-pos(0,1))); // bottom
    }

    if (p0 != 6) {
        if (!(board[p0+pos(1,0)] & color)) rook_moves.insert(move(p0, p0+pos(1,0))); // right
    }

    for (int x=getx(p0)-1; x>=0; x--) {
        U8 p1 = pos(x, gety(p0));
        if (board[p1]) {
            refl_blocked = true;
            if (board[p1] & color) break;         // our piece
            else rook_moves.insert(move(p0, p1)); // their piece - capture
            break;
        }
        else {
            rook_moves.insert(move(p0, p1));
        }
    }

    if (refl_blocked) return rook_moves;
    
    if (p0 < 8) {
        for (int p1 : left_rook_reflect) {
            if (board[p1]) {
                if (board[p1] & color) break;         // our piece
                else rook_moves.insert(move(p0, p1)); // their piece - capture
                break;
            }
            else {
                rook_moves.insert(move(p0, p1));
            }
        }
    }

    return rook_moves;
}

std::unordered_set<U16> construct_bottom_bishop_moves_with_board(const U8 p0, const U8* board) {

    PlayerColor color = color(board[p0]);
    std::unordered_set<U16> bishop_moves;
    // top right - move back
    if (p0 < 6 || p0 == 13) {
        if (!(board[p0+pos(0,1)+pos(1,0)] & color)) bishop_moves.insert(move(p0, p0+pos(0,1)+pos(1,0)));
    }
    // bottom right - move back
    if (p0 > 6) 
    {
        if (!(board[p0-pos(0,1)+pos(1,0)] & color)) bishop_moves.insert(move(p0, p0-pos(0,1)+pos(1,0)));
    }
    std::vector<U8> p1s;
    std::vector<U8> p1s_2;
    // top left - forward / reflections
    if (p0 == 1) {
        p1s.push_back(pos(0,1));
        p1s.push_back(pos(1,2));
    }
    else if (p0 == 2) {
        p1s.push_back(pos(1,1));
        p1s.push_back(pos(0,2));
        p1s.push_back(pos(1,3));
    }
    else if (p0 == 3) {
        p1s.push_back(pos(2,1));
        p1s.push_back(pos(1,2));
        p1s.push_back(pos(0,3));
        p1s.push_back(pos(1,4));
        p1s.push_back(pos(2,5));
        p1s.push_back(pos(3,6));
    }
    else if (p0 == 4 || p0 == 5) {
        p1s.push_back(p0-pos(2,0));
        p1s.push_back(p0+pos(0,1)-pos(1,0));
    }
    else if (p0 == 6) {
        p1s.push_back(pos(5,1));
    }
    else if (p0 == 10) {
        p1s_2.push_back(pos(1,0));
        p1s_2.push_back(pos(0,1));

        p1s.push_back(pos(1,2));
        p1s.push_back(pos(0,3));
        p1s.push_back(pos(1,4));
        p1s.push_back(pos(2,5));
        p1s.push_back(pos(3,6));
    }
    else if (p0 == 11) {
        p1s.push_back(pos(2,0));
        p1s.push_back(pos(1,1));
        p1s.push_back(pos(0,2));
    }
    else if (p0 == 12) {
        p1s.push_back(pos(3,0));
        p1s.push_back(pos(2,1));
        p1s.push_back(pos(1,2));
        p1s.push_back(pos(0,3));
    }
    else if (p0 == 13) {
        p1s.push_back(pos(4,0));
        p1s.push_back(pos(3,1));
    }

    // back 
    if (p0 < 6 || p0 >= 12) {
        p1s.push_back(pos(getx(p0)+1,gety(p0)+1));
    }
    if (p0 > 9) {
        p1s.push_back(pos(getx(p0)+1,gety(p0)-1));
    }

    for (auto p1 : p1s) {
        if (board[p1]) {
            if (board[p1] & color) break;           // our piece
            else bishop_moves.insert(move(p0, p1)); // their piece - capture
            break;
        }
        else {
            bishop_moves.insert(move(p0, p1));
        }
    }

    for (auto p1 : p1s_2) {
        if (board[p1]) {
            if (board[p1] & color) break;           // our piece
            else bishop_moves.insert(move(p0, p1)); // their piece - capture
            break;
        }
        else {
            bishop_moves.insert(move(p0, p1));
        }
    }

    return bishop_moves;
}

std::unordered_set<U16> construct_bottom_pawn_moves_with_board(const U8 p0, const U8 *board, bool promote = false) {
    
    PlayerColor color = color(board[p0]);
    std::unordered_set<U16> pawn_moves;

    if (!(board[pos(getx(p0)-1,0)] & color)) {
        if (promote) {
            pawn_moves.insert(move_promo(p0, pos(getx(p0)-1,0), PAWN_ROOK));
            pawn_moves.insert(move_promo(p0, pos(getx(p0)-1,0), PAWN_BISHOP));
        }
        else {
            pawn_moves.insert(move(p0, pos(getx(p0)-1,0)));
        }
    }
    if (!(board[pos(getx(p0)-1,1)] & color)) {
        if (promote) {
            pawn_moves.insert(move_promo(p0, pos(getx(p0)-1,1), PAWN_ROOK));
            pawn_moves.insert(move_promo(p0, pos(getx(p0)-1,1), PAWN_BISHOP));
        }
        else {
            pawn_moves.insert(move(p0, pos(getx(p0)-1,1)));
        }
    }
    if (p0 == 10 && !(board[17] & color)) pawn_moves.insert(move(p0, 17));

    return pawn_moves;
}

std::unordered_set<U16> construct_bottom_king_moves_with_board(const U8 p0, const U8 *board) {

    // king can't move into check. See if these squares are under threat from 
    // enemy pieces as well.
    PlayerColor color = color(board[p0]);
    std::unordered_set<U16> king_moves;
    if (!(board[pos(getx(p0)-1,0)] & color)) king_moves.insert(move(p0, pos(getx(p0)-1,0)));
    if (!(board[pos(getx(p0)-1,1)] & color)) king_moves.insert(move(p0, pos(getx(p0)-1,1)));
    if (p0 == 10 && !(board[pos(getx(p0)-1,2)] & color)) king_moves.insert(move(p0, pos(getx(p0)-1,2)));
    if (p0 != 6 && !(board[pos(getx(p0)+1,0)] & color)) king_moves.insert(move(p0, pos(getx(p0)+1,0)));
    if (p0 != 6 && !(board[pos(getx(p0)+1,1)] & color)) king_moves.insert(move(p0, pos(getx(p0)+1,1)));
    if (p0 >= 12 && !(board[pos(getx(p0)+1,2)] & color)) king_moves.insert(move(p0, pos(getx(p0)+1,2)));
    if (p0 == 13 && !(board[pos(getx(p0),2)] & color)) king_moves.insert(move(p0, pos(getx(p0),2)));
    if (!(board[pos(getx(p0),gety(p0)^1)] & color)) king_moves.insert(move(p0, pos(getx(p0),gety(p0)^1)));

    return king_moves;
}

char piece_to_char(U8 piece) {
    char ch = '.';
    if      (piece & PAWN)   ch = 'p';
    else if (piece & ROOK)   ch = 'r';
    else if (piece & BISHOP) ch = 'b';
    else if (piece & KING)   ch = 'k';

    if (piece & WHITE) ch = ch - ('a'-'A');

    return ch;
}

std::string board_to_str(const U8 *board) {

    std::string board_str = ".......\n.......\n..   ..\n..   ..\n..   ..\n.......\n.......\n";

    for (int i=0; i<56; i++) {
        U8 piece = board[i];
        if (board_str[i] == '\n' || board_str[i] == ' ') continue;
        board_str[(48-(i/8)*8) + i%8] = piece_to_char(piece);
    }

    return board_str;
}

std::string player_to_play_to_str(const Board& b) {
    if (b.data.player_to_play == WHITE) {
        return "WHITE";
    }
    else if (b.data.player_to_play == BLACK) {
        return "BLACK";
    }
    else {
        return "UNKNOWN";
    }
}

std::string all_boards_to_str(const Board& b) {
    std::string board_str(256, ' ');
    std::string board_mask = ".......\n.......\n..   ..\n..   ..\n..   ..\n.......\n.......\n";
    const U8 (*boards)[64] = &(b.data.board_0);
    for (int b=0; b<4; b++) {
        for (int i=0; i<56; i++) {
            if (board_mask[i] == '\n' || board_mask[i] == ' ') continue;
            board_str[(224-(i/8)*32) + b*8 + i%8] = piece_to_char(boards[b][i]);
        }
    }
    for (int i=31; i<256; i+=32) {
        board_str[i] = '\n';
    }
    return board_str.substr(32);
}

std::string move_to_str(U16 move) {

    std::string s = "a1a1";
    s[0] += getx(getp0(move));
    s[1] += gety(getp0(move));
    s[2] += getx(getp1(move));
    s[3] += gety(getp1(move));
    if (getpromo(move) & PAWN_BISHOP) {
        s += "b";
    }
    else if (getpromo(move) & PAWN_ROOK) {
        s += "r";
    }
    return s;
}

U16 str_to_move(std::string move) {
    
    U8 x0 = move[0] - 'a';
    U8 y0 = move[1] - '1';
    U8 x1 = move[2] - 'a';
    U8 y1 = move[3] - '1';
    U8 promo = 0;
    if (move.size() > 4) {
        if (move[4] == 'r') promo = PAWN_ROOK;
        else promo = PAWN_BISHOP;
    }

    return move_promo(pos(x0,y0), pos(x1,y1), promo);
}

std::unordered_set<U16> p_get_pseudolegal_moves_for_piece(U8 piece_pos, Board* b) {

    std::unordered_set<U16> moves;
    U8 piece_id = b->data.board_0[piece_pos];

    std::unordered_set<U8> bottom({ 1, 2, 3, 4, 5, 6, 10, 11, 12, 13 });
    std::unordered_set<U8> left({ 0, 8, 16, 24, 32, 40, 9, 17, 25, 33 });
    std::unordered_set<U8> top({ 48, 49, 50, 51, 52, 53, 41, 42, 43, 44 });
    std::unordered_set<U8> right({ 54, 46, 38, 30, 22, 14, 45, 37, 29, 21 });

    const U8 *board = b->data.board_0;
    const U8 *coord_map = id;
    const U8 *inv_coord_map = id;
    if      (left.count(piece_pos))  { board = b->data.board_270;  coord_map = acw_90; inv_coord_map = cw_90;  }
    else if (top.count(piece_pos))   { board = b->data.board_180; coord_map = cw_180; inv_coord_map = cw_180; }
    else if (right.count(piece_pos)) { board = b->data.board_90; coord_map = cw_90;  inv_coord_map = acw_90; }

    if (piece_id & PAWN) {
        if (((piece_pos == 51 || piece_pos == 43) && (piece_id & WHITE)) || 
            ((piece_pos == 11 || piece_pos == 3)  && (piece_id & BLACK)) ) {
            moves = construct_bottom_pawn_moves_with_board(coord_map[piece_pos], board, true);
        }
        else {
            moves = construct_bottom_pawn_moves_with_board(coord_map[piece_pos], board);
        }
    }
    else if (piece_id & ROOK) {
        moves = construct_bottom_rook_moves_with_board(coord_map[piece_pos], board);
    }
    else if (piece_id & BISHOP) {
        moves = construct_bottom_bishop_moves_with_board(coord_map[piece_pos], board);
    }
    else if (piece_id & KING) {
        moves = construct_bottom_king_moves_with_board(coord_map[piece_pos], board);
    }

    moves = transform_moves(moves, inv_coord_map);

    return moves;
}

void rotate_board(U8 *src, U8 *tgt, const U8 *transform) {

    for (int i=0; i<64; i++) {
        tgt[transform[i]] = src[i];
    }
}
std::unordered_set<U16> p_get_pseudolegal_moves_for_side(U8 color,Board* b) {

    // std::cout << "Getting Pseudolegal moves for " << (char)((color>>5) + 'a') << "\n";
    std::unordered_set<U16> pseudolegal_moves;

    U8 *pieces = (U8*)(&(b->data));

    if (color == WHITE) {
        pieces = pieces + 6;
    }

    for (int i=0; i<6; i++) {
        //std::cout << "checking " << piece_to_char(b->data.board_0[pieces[i]]) << "\n";
        if (pieces[i] == DEAD) continue;
        //std::cout << "Getting Moves for " << piece_to_char(b->data.board_0[pieces[i]]) << "\n";
        auto piece_moves = p_get_pseudolegal_moves_for_piece(pieces[i],b);
        pseudolegal_moves.insert(piece_moves.begin(), piece_moves.end());
    }

    return pseudolegal_moves;

}
// Optimization: generate inverse king moves
// For now, just generate moves of the opposite color and check if any of them
// attack the king square
bool p_under_threat(U8 piece_pos,Board* b){

    auto pseudolegal_moves = private_code::p_get_pseudolegal_moves_for_side(b->data.player_to_play ^ (WHITE | BLACK),b);

    for (auto move : pseudolegal_moves) {
        // std::cout << move_to_str(move) << " ";
        if (getp1(move) == piece_pos) {
            // std::cout << "<- causes check\n";
            return true;
        }
    }
    // std::cout << std::endl;

    return false;
}

bool pin_check(Board* b) {

    auto king_pos = b->data.w_king;
    // can make b branchless for kicks but won't add much performance
    if (b->data.player_to_play == BLACK) {
        king_pos = b->data.b_king;
    }

    return p_under_threat(king_pos,b);
}

std::unordered_set<U16> p_get_pseudolegal_moves(Board* b) {
    return p_get_pseudolegal_moves_for_side(b->data.player_to_play,b);
}



// legal move generation:
// if the king is under check: (Possible optimization, as b should be faster)
//     Look for moves that protect the king from check
// else:
//     Get all pseudolegal moves
//     for each pseudolegal move for our color:
//         if doing b move will leave the king in threat from opponent's pieces
//             don't add the move to legal moves
//         else
//             add to legal moves
//
// Only implement the else case for now
void p_flip_player(Board* b) {
    b->data.player_to_play = (PlayerColor)(b->data.player_to_play ^ (WHITE | BLACK));
}

void p_do_move(U16 move,Board* b) {

    U8 p0 = getp0(move);
    U8 p1 = getp1(move);
    U8 promo = getpromo(move);

    U8 piecetype = b->data.board_0[p0];
    b->data.last_killed_piece = 0;
    b->data.last_killed_piece_idx = -1;

    // scan and get piece from coord
    U8 *pieces = (U8*)b;
    for (int i=0; i<12; i++) {
        if (pieces[i] == p1) {
            pieces[i] = DEAD;
            b->data.last_killed_piece = b->data.board_0[p1];
            b->data.last_killed_piece_idx = i;
        }
        if (pieces[i] == p0) {
            pieces[i] = p1;
        }
    }

    if (promo == PAWN_ROOK) {
        piecetype = (piecetype & (WHITE | BLACK)) | ROOK;
    }
    else if (promo == PAWN_BISHOP) {
        piecetype = (piecetype & (WHITE | BLACK)) | BISHOP;
    }

    b->data.board_0[p1]           = piecetype;
    b->data.board_90[cw_90[p1]]   = piecetype;
    b->data.board_180[cw_180[p1]] = piecetype;
    b->data.board_270[acw_90[p1]] = piecetype;

    b->data.board_0[p0]           = 0;
    b->data.board_90[cw_90[p0]]   = 0;
    b->data.board_180[cw_180[p0]] = 0;
    b->data.board_270[acw_90[p0]] = 0;

    // std::cout << "Did last move\n";
    // std::cout << all_boards_to_str(*b);
}

void p_undo_last_move(U16 move,Board * b) {

    U8 p0 = getp0(move);
    U8 p1 = getp1(move);
    U8 promo = getpromo(move);

    U8 piecetype = b->data.board_0[p1];
    U8 deadpiece = b->data.last_killed_piece;
    b->data.last_killed_piece = 0;

    // scan and get piece from coord
    U8 *pieces = (U8*)(&(b->data));
    for (int i=0; i<12; i++) {
        if (pieces[i] == p1) {
            pieces[i] = p0;
            break;
        }
    }
    if (b->data.last_killed_piece_idx >= 0) {
        pieces[b->data.last_killed_piece_idx] = p1;
        b->data.last_killed_piece_idx = -1;
    }

    if (promo == PAWN_ROOK) {
        piecetype = ((piecetype & (WHITE | BLACK)) ^ ROOK) | PAWN;
    }
    else if (promo == PAWN_BISHOP) {
        piecetype = ((piecetype & (WHITE | BLACK)) ^ BISHOP) | PAWN;
    }

    b->data.board_0[p0]           = piecetype;
    b->data.board_90[cw_90[p0]]   = piecetype;
    b->data.board_180[cw_180[p0]] = piecetype;
    b->data.board_270[acw_90[p0]] = piecetype;

    b->data.board_0[p1]           = deadpiece;
    b->data.board_90[cw_90[p1]]   = deadpiece;
    b->data.board_180[cw_180[p1]] = deadpiece;
    b->data.board_270[acw_90[p1]] = deadpiece;

    // std::cout << "Undid last move\n";
    // std::cout << all_boards_to_str(*b);
}


}

#pragma endregion
#pragma region evaluation_functions
vector<int> positional_advantage_rook = {0,3,1,1,1,1,0,0,0,0};
double rook_positional_advantage_for_piece(U8 piece) { 
    std::unordered_set<U8> bottom({ 1, 2, 3, 4, 5, 6, 10, 11, 12, 13 }); 
    std::unordered_set<U8> left({ 0, 8, 16, 24, 32, 40, 9, 17, 25, 33 }); 
    std::unordered_set<U8> top({ 48, 49, 50, 51, 52, 53, 41, 42, 43, 44 }); 
    std::unordered_set<U8> right({ 54, 46, 38, 30, 22, 14, 45, 37, 29, 21 }); 
    if(bottom.count(piece)) return positional_advantage_rook[piece]; 
    else if (left.count(piece)) return positional_advantage_rook[private_code::acw_90[piece]]; 
    else if (top.count(piece)) return positional_advantage_rook[private_code::cw_180[piece]]; 
    else return positional_advantage_rook[private_code::cw_90[piece]];
    
}

double bishop_positional_advantage_for_piece(U8 piece)
{
    int x = getx(piece);
    int y = gety(piece);
    if(x + y == 3 || x + y == 9 || x-y == -3 || x-y == 3) return 1;
    return 0;
}
double bishop_positional_advantage(Board& b, bool for_white)
{
    double score = 0;
    U8 *pieces = (U8*) &(b.data);
    if(for_white) pieces += 6; //b is to get the white pieces.
    score += bishop_positional_advantage_for_piece(pieces[3]);
    if(b.data.board_0[pieces[4]] & BISHOP) score += bishop_positional_advantage_for_piece(pieces[4]);
    if(b.data.board_0[pieces[5]] & BISHOP) score += bishop_positional_advantage_for_piece(pieces[5]);
    return score;
}

double rook_positional_advantage(Board &b, bool for_white)
{
    // pawns can become rooks
    double score = 0;
    U8 *pieces = (U8*) &(b.data);
    if(for_white) pieces += 6; //b is to get the white pieces.
    for(int i = 0; i < 6; i++)
    {
        if(pieces[i] == DEAD) continue;
        U8 piece = b.data.board_0[pieces[i]];
        //if(piece & (for_white ? WHITE : BLACK))
        {
            if((piece & ROOK) != 0) score += rook_positional_advantage_for_piece(pieces[i]);
        }
    }
    return score;
}
double safe_advantage(Board&b, bool for_white)
{
    unordered_set<U16> enemy_moves = private_code::p_get_pseudolegal_moves_for_side(for_white ? BLACK : WHITE, &b);
    for(auto move : enemy_moves)
    {
        if(getp1(move) & (for_white ? WHITE : BLACK)) return 0;
    }
    return 1;
}
double fpoints(Board &b, bool for_white)
{
    double score = 0;
    U8 *pieces = (U8*) &(b.data);
    if(for_white) pieces += 6; //b is to get the white pieces.
    for(int i = 0; i < 6; i++)
    {
        if(pieces[i] == DEAD) continue;
        U8 piece = b.data.board_0[pieces[i]];
        //if(piece & (for_white ? WHITE : BLACK))
        {
            if((piece & PAWN) != 0) score += 1;
            else if((piece & ROOK) != 0) score += 5; //we keep more score for rook since it is more important to us.
            else if((piece & BISHOP) != 0) score += 3;
            else if((piece & KING) != 0) continue;; //there should technically be no score for the king
        }
    }
    return score;
}
double fmoves_for_pawn( Board &b, bool for_white)
{   
    U8 loc1 = b.data.w_pawn_ws;
    U8 loc2 = b.data.w_pawn_bs;
    if(for_white == false)
    {
        loc1 = b.data.b_pawn_ws;
        loc2 = b.data.b_pawn_bs;
    }
    if(loc1 == DEAD && loc2 == DEAD) //if both rooks are dead, then we return 0.
    {
        return 0;
    }
    else if(loc1 == DEAD)
    {
        return private_code::p_get_pseudolegal_moves_for_piece(loc2,&b).size();
    }
    else if(loc2 == DEAD)
    {
        return private_code::p_get_pseudolegal_moves_for_piece(loc1,&b).size();
    }
    int moves_available = private_code::p_get_pseudolegal_moves_for_piece(loc1,&b).size() + private_code::p_get_pseudolegal_moves_for_piece(loc2,&b).size();
    return moves_available;
}
double fmoves_for_rook( Board &b, bool for_white)
{   
    vector<U8> locations;
    if(for_white)
    {
        if(b.data.w_rook_bs != DEAD) locations.push_back(b.data.w_rook_bs);
        if(b.data.w_rook_ws != DEAD) locations.push_back(b.data.w_rook_ws);
        if(b.data.w_pawn_ws != DEAD && b.data.board_0[b.data.w_pawn_ws] & ROOK) locations.push_back(b.data.w_pawn_ws);
        if(b.data.w_pawn_bs != DEAD && b.data.board_0[b.data.w_pawn_bs] & ROOK) locations.push_back(b.data.w_pawn_bs);
    }
    if(for_white == false)
    {
        if(b.data.b_rook_bs != DEAD) locations.push_back(b.data.b_rook_bs);
        if(b.data.b_rook_ws != DEAD) locations.push_back(b.data.b_rook_ws);
        if(b.data.b_pawn_ws != DEAD && b.data.board_0[b.data.b_pawn_ws] & ROOK) locations.push_back(b.data.b_pawn_ws);
        if(b.data.b_pawn_bs != DEAD && b.data.board_0[b.data.b_pawn_bs] & ROOK) locations.push_back(b.data.b_pawn_bs);
    }
    if(locations.size() == 0) //if all rooks are dead, then we return 0.
    {
        return 0;
    }
    int moves_available = 0; 
    for (auto loc : locations)
    {
        moves_available += private_code::p_get_pseudolegal_moves_for_piece(loc,&b).size();
    }
    return moves_available;
}
double fmoves_for_bishop( Board &b, bool for_white)
{   
    vector<U8> locations;
    if(for_white)
    {
        if(b.data.w_bishop != DEAD) locations.push_back(b.data.w_bishop);
        if(b.data.w_pawn_ws != DEAD && b.data.board_0[b.data.w_pawn_ws] & BISHOP) locations.push_back(b.data.w_pawn_ws);
        if(b.data.w_pawn_bs != DEAD && b.data.board_0[b.data.w_pawn_bs] & BISHOP) locations.push_back(b.data.w_pawn_bs);
    }
    if(for_white == false)
    {
        if(b.data.b_bishop != DEAD) locations.push_back(b.data.b_bishop);
        if(b.data.b_pawn_ws != DEAD && b.data.board_0[b.data.b_pawn_ws] & BISHOP) locations.push_back(b.data.b_pawn_ws);
        if(b.data.b_pawn_bs != DEAD && b.data.board_0[b.data.b_pawn_bs] & BISHOP) locations.push_back(b.data.b_pawn_bs);
    }
    if(locations.size() == 0) //if all rooks are dead, then we return 0.
    {
        return 0;
    }
    int moves_available = 0; 
    for (auto loc : locations)
    {
        moves_available += private_code::p_get_pseudolegal_moves_for_piece(loc,&b).size();
    }
    return moves_available;
}
double fmoves_for_king( Board &b, bool for_white)
{
    U8 loc1 = b.data.w_king;
    if(for_white == false)
    {
        loc1 = b.data.b_king;
    }
    unordered_set<U16> available_moves = private_code::p_get_pseudolegal_moves_for_piece(loc1,&b);
    int total_safe_moves = 0; 
    Board *bcopy = b.copy();
    for(auto move : available_moves)
    {
        bcopy->do_move(move); private_code::p_flip_player(bcopy); //b switches back to us.
        if(bcopy->in_check() == false)
        {
            total_safe_moves++;
        }
        private_code::p_undo_last_move(move,bcopy);  //undos the move.
    }
    delete bcopy;
    //possibly non-linear, so we return a power of our available moves, as if b is high, it doesn't matter at all.
    double result = pow((double)total_safe_moves,(double)0.7);
    return result;
}
double in_check(Board& b, bool for_white) //(negative returned, -1 if in check, 0 if not in check)
{
    PlayerColor cur_player = (for_white ? WHITE : BLACK); 
    if(cur_player != b.data.player_to_play) return 0; //not even their chance, how can they be in check.
    //else
    if(b.in_check()) 
    {
        return cur_player == our_color ? (double) -1 :(double) 1; 
    }
    else return 0;
}
inline U8 get_king_pos( Board& b, bool player)
{
    if(player) return b.data.w_king;
    else return b.data.b_king;
}
double distance_pawn_activation(int dist)
{
    dist += 2; //so now everything should be >= 2.
    //now being a rook is 0, a bishop is 1, a pawn is distance + 2, and being a dead piece is 10.
    //there is a huge jump from dying and being a pawn, so we need to make b somewhat non-linear.
    //we need to allow it to die when it can attack and steal an important piece from the opponent.
    //hence its weight must be much lower than that of a rook or a bishop.
    return (10 - dist); 
}
double promotion_advantage(Board& b, bool for_white)
{
    U8 pawns[2];
    if(for_white)
    {
        pawns[0] = b.data.w_pawn_ws;
        pawns[1] = b.data.w_pawn_bs;
    }
    else
    {
        pawns[0] = b.data.b_pawn_ws;
        pawns[1] = b.data.b_pawn_bs;
    }
    U8 pieces[2];
    pieces[0] = b.data.board_0[pawns[0]];
    pieces[1] = b.data.board_0[pawns[1]];
    int dists[2] = {0,0};
    for(int i = 0; i < 2; i++)
    {
        if(pawns[i] == DEAD)
        {
            dists[i] = 8;
            continue;
        }
        if(pieces[i] & PAWN)
        {
            //else it is not a pawn and we give it some benefit for not being a pawn anymore.
            int posx = getx(pawns[i]); int posy(gety(pawns[i])); 
            if(for_white)
            {
                dists[i] = max(5 - posy,0) + 3;
                if(posy >= 4)
                {
                    dists[i] += (-3) + max(0,4 - posx);
                }
            }
            else
            {
                dists[i] = max(0,posy-1) + 3;
                if(posy <= 2)
                {
                    dists[i] += (-3) + max(0,posx - 2); 
                }
            }
        }   
        else if(pieces[i] & ROOK)
        {
            dists[i] = -2; 
        }
        else if(pieces[i] & BISHOP)
        {
            dists[i] = -1;
        }
    }
    return distance_pawn_activation(dists[0]) + distance_pawn_activation(dists[1]);
}
double fweighted_points_diff(Board &b, bool for_white)
{
    double score = fpoints(b, for_white);
    //now we do some transformation to this score, so that difference in points is not linear.
    //the lower the absolute score, the more the difference in points matters.
    //if the score is high, then the difference in points doesn't matter much, but if the absolute score is lower, then difference in points matters much more.
    //we want to make the score non-linear, so we do a power of the score.
    score = pow(score, 0.8);
    return score;
}
double under_attack(Board& b, bool is_white) // U8 represents piece type which is a 7 bit COLOR | PIECETYPE
{
//   unordered_map<U8,U8> attacker_to_attackee;
    double weight = 0;
    U8* pieces = (U8*) &(b.data);
    if(!is_white) pieces += 6; //b is to get the white pieces.
    for(int i = 0 ; i < 6; i++)
    {
        unordered_set<U16> enemy_moves = private_code::p_get_pseudolegal_moves_for_piece(pieces[i],&b);
        for(auto move : enemy_moves)
        {
            U8 attacker_position = getp0(move);
            U8 attackee_position = getp1(move);
            U8 attackee = b.data.board_0[attackee_position];
            U8 attacker = b.data.board_0[attacker_position];
            if(attackee & (is_white ? WHITE : BLACK))
            {
//                attacker_to_attackee[attacker] = attackee;
                if(attackee & PAWN) weight -= 1;
                else if(attackee & ROOK) weight -= 5;
                else if(attackee & BISHOP) weight -= 3;
                private_code::p_do_move(move,&b); 
                std::unordered_set<U16> our_moves = private_code::p_get_pseudolegal_moves_for_side(is_white ? WHITE : BLACK,&b);
                for(auto our_move : our_moves)
                {
                    if(getp1(our_move) == attackee_position)
                    {
                        if(attacker & PAWN) {
                            weight += 1;
                            break;
                        }
                        else if(attacker & ROOK) {
                            weight += 5;
                            break;
                        }
                        else if(attacker & BISHOP) {
                            weight += 3;
                            break;
                        }
                    }
                }
                private_code::p_undo_last_move(move,&b);
            }
        }
    }
    double our_turn_lowering = 0.5;
    if((is_white ? WHITE : BLACK) == b.data.player_to_play) return our_turn_lowering * weight;
    else return weight;
}
double pawn_attack(Board& b, bool is_white)
{
    U8 pawns[2];
    if(is_white)
    {
        pawns[0] = b.data.b_pawn_bs;
        pawns[1] = b.data.b_pawn_ws;
    }
    else 
    {
        pawns[0] = b.data.w_pawn_bs;
        pawns[1] = b.data.w_pawn_ws;
    }
    double weight = 0;
    for(int i = 0 ; i < 2 ; i++)
    {
        unordered_set<U16> enemy_moves = private_code::p_get_pseudolegal_moves_for_piece(pawns[i],&b);
        for(auto move : enemy_moves)
        {
            U8 attacker_position = getp0(move);
            U8 attackee_position = getp1(move);
            U8 attackee = b.data.board_0[attackee_position];
            U8 attacker = b.data.board_0[attacker_position];
            if(attackee & (is_white ? WHITE : BLACK))
            {
                if(attackee & PAWN) weight -= 1;
                else if(attackee & ROOK) weight -= 5;
                else if(attackee & BISHOP) weight -= 3;
                private_code::p_do_move(move,&b); 
                std::unordered_set<U16> our_moves = private_code::p_get_pseudolegal_moves_for_side(is_white ? WHITE : BLACK,&b);
                for(auto our_move : our_moves)
                {
                    if(getp1(our_move) == attackee_position)
                    {
                        if(attacker & PAWN) {
                            weight += 1;
                            break;
                        }
                        else if(attacker & ROOK) {
                            weight += 5;
                            break;
                        }
                        else if(attacker & BISHOP) {
                            weight += 3;
                            break;
                        }
                    }
                }
                private_code::p_undo_last_move(move,&b);

            }
        }
    }
    return weight;
}
double rook_attack(Board& b, bool is_white)
{
    U8 pawns[2];
    if(is_white)
    {
        pawns[0] = b.data.b_pawn_bs;
        pawns[1] = b.data.b_pawn_ws;
    }
    else 
    {
        pawns[0] = b.data.w_pawn_bs;
        pawns[1] = b.data.w_pawn_ws;
    }
       double weight = 0;
       for(int i = 0 ; i < 2 ; i++){
       unordered_set<U16> enemy_moves = private_code::p_get_pseudolegal_moves_for_piece(pawns[i],&b);
       for(auto move : enemy_moves)
        {
            U8 attacker_position = getp0(move);
            U8 attackee_position = getp1(move);
            U8 attackee = b.data.board_0[attackee_position];
            U8 attacker = b.data.board_0[attacker_position];
            if(attackee & (is_white ? WHITE : BLACK))
            {
                if(attackee & PAWN) weight -= 1;
                else if(attackee & ROOK) weight -= 5;
                else if(attackee & BISHOP) weight -= 3;
                private_code::p_do_move(move,&b);
                std::unordered_set<U16> our_moves = private_code::p_get_pseudolegal_moves_for_side(is_white ? WHITE : BLACK,&b);
                for(auto our_move : our_moves)
                {
                    if(getp1(our_move) == attackee_position)
                    {
                        weight += 5;
                    }
                }
                private_code::p_undo_last_move(move,&b);


            }
        }
       }
       return weight;
}
#pragma endregion
#pragma region weights
std::unordered_map<string,double> weight_map = {{"weighted_points_diff", 0},{"under_attack", 5.094},{"in_check", 2.4127},{"points",13.906}, {"moves_for_pawn",1.03}, {"moves_for_rook",1.883}, {"moves_for_bishop",0.0495}, {"moves_for_king",0.0007}, {"promotion_advantage", 3.392},{"rook_positional_advantage",0},{"bishop_positional_advantage",0},{"safe_advantage",0}};
std::unordered_map<string,double> untrained_weight_map = {{"weighted_points_diff", 0},{"under_attack", 5.094},{"in_check", 2.4127},{"points",13.906}, {"moves_for_pawn",1.03}, {"moves_for_rook",1.883}, {"moves_for_bishop",0.0495}, {"moves_for_king",0.0007}, {"promotion_advantage", 3.392},{"rook_positional_advantage",0},{"bishop_positional_advantage",0},{"safe_advantage",0}};
//std::unordered_map<string,double> untrained_weight_map = {{"under_attack",3.23599},{"in_checkmate", 2},{"in_check", 1.12759},{"points",10}, {"moves_for_pawn",0.556186}, {"moves_for_rook",1.47579}, {"moves_for_bishop",0.00637268}, {"moves_for_king",0.289523},{"promotion_advantage",  1.37812}};
std::unordered_map<string,feature> feature_map = {{"weighted_points_diff", fweighted_points_diff},{"under_attack", under_attack},{"in_check",in_check},{"points", fpoints}, {"moves_for_pawn",fmoves_for_pawn}, {"moves_for_rook",fmoves_for_rook}, {"moves_for_bishop",fmoves_for_bishop},{"promotion_advantage", promotion_advantage} ,{"moves_for_king",fmoves_for_king},{"rook_positional_advantage",rook_positional_advantage},{"bishop_positional_advantage",bishop_positional_advantage},{"safe_advantage",safe_advantage}}; 
//{"under_attack", under_attack},
void get_weights_from_file(string filename)
{
    if(!qmode) filename =  filename  + (should_update_weights >= 1? "up" : "" ) + ".txt";
    else filename = filename + (should_update_weights >= 1? "up" : "" ) + "q.txt";
    ifstream infile(filename); 
    string line; 
    while(getline(infile,line))
    {
        string feature_name; double weight;
        stringstream ss(line);
        ss >> feature_name >> weight;
        weight_map[feature_name] = weight;
    }
    infile.close();
    infile.open("weights/wuntrained.txt"); 
    line = ""; 
    while(getline(infile,line))
    {
        string feature_name; double weight;
        stringstream ss(line);
        ss >> feature_name >> weight;
        untrained_weight_map[feature_name] = weight;
      //  cout << "got feature name , weight = " << feature_name << " " << weight << endl;
    }
     //cout << "got weights for as follows: " << endl;
     //if(should_update_weights){
     //for(auto p : weight_map)
     //{
     //    cout << p.first << " " << p.second << endl;
     //}
     //}
     //else{
     //   for(auto p : untrained_weight_map)
     //   {
     //       cout << p.first << " " << p.second << endl;
     //   }
     //}
     //cout << endl; 
}
void write_weights_to_file(string filename)
{
    if(!qmode) filename =  filename  + (should_update_weights >= 1? "up" : "" ) + ".txt";
    else filename = filename + (should_update_weights >= 1? "up" : "" ) + "q.txt";
    ofstream outfile(filename);
    for(auto p : weight_map)
    {
        outfile << p.first << " " << p.second << endl;
    }
}
void update_weights(Board &b, double actual_cost ,int at_depth)
{
    if(should_update_weights == 0) return;
    double predicted_cost = evaluate(b,feature_map,weight_map);
    double error = actual_cost - predicted_cost;
    bool we_are_white = (our_color == WHITE);
    unordered_map<string,double> weight_updates;
    double total = 0;
    for(auto &p : weight_map)
    {
        if(feature_map.find(p.first) == feature_map.end())
        {
            cout << "missed a parameter function" << endl;
            continue;
        }
        weight_updates[p.first] = pow(at_depth,2) * learning_rate * error * (p.second > 0 ? 1 : -1)  * ((feature_map[p.first])(b,we_are_white) - feature_map[p.first](b,!we_are_white)); 
        total += weight_updates[p.first] * weight_updates[p.first]; //(weight_updates[p.first]);
    }
    if(debugmode)
        cout << "reached here at depth " << at_depth << endl;
    double norm = sqrt(total);
    //now we decrease b by the step function that we have (learning rate).
    norm = norm/learning_rate; //b is the norm of the vector.
    for(auto &p : weight_map)
    {
        double val =(norm == 0 ? 0 : weight_updates[p.first]/norm);
        if(p.second + val > 0)
        {
            p.second += val;
        }
    }
    write_weights_to_file();
    if(debugmode)
    {
        cout << "\n";
    }
}
void update_weights_delta(Board &b, double actual_cost ,int at_depth, U8 move)
{
    if(should_update_weights == 0) return;
    double predicted_cost = evaluate(b,feature_map,weight_map);
    double error = actual_cost - predicted_cost;
    bool we_are_white = (our_color == WHITE);
    unordered_map<string,double> weight_updates;
    double total = 0;
    Board *bcopy = b.copy();
    for(auto &p : weight_map)
    {
        weight_updates[p.first] = at_depth * learning_rate * error * (p.second > 0 ? 1 : -1)  * (-1*((feature_map[p.first])(b,we_are_white) - feature_map[p.first](b,!we_are_white)) + (feature_map[p.first](*bcopy,we_are_white) - feature_map[p.first](*bcopy, !we_are_white))); 
        total += weight_updates[p.first] * weight_updates[p.first]; //(weight_updates[p.first]);
    }
    delete bcopy;
    cout << "reached here at depth " << at_depth << endl;
    double norm = sqrt(total);
    //now we decrease b by the step function that we have (learning rate).
    norm = norm/learning_rate; //b is the norm of the vector.
    for(auto &p : weight_map)
    {
        double val =(norm == 0 ? 0 : weight_updates[p.first]/norm);
        if(p.second + val > 0)
        {
            p.second += val;
            if(debugmode)
            {
                cout << p.first << "+= " << val << " to " << p.second << "\n";
            }
        }
    }
    write_weights_to_file("weights/new_feature.txt");
    if(debugmode)
    {
        cout << "\n";
    }
}
#pragma endregion
#pragma region improvements
//setting a dynamic depth. DONE
//setting a time limit. DONE
//ordering the moves in the best way possible for alpha beta pruning. DONE
//Adding a simple quaiscence search. DONE
//Adding a detection for 3fold repetition. NOT DONE
//for now ordering can be implemented through a simple evaluation at current point heuristic.
#pragma endregion
pair<U16,bool> capture_possible(Board& b)
{
    //returns true if a capture is possible.
    unordered_set<U16> moves = b.get_legal_moves();
    PlayerColor cur_player = b.data.player_to_play;
    bool cur_player_is_white = (cur_player == WHITE);
    for(auto move : moves)
    {
        if(b.data.board_0[getp1(move)] & (cur_player_is_white ? BLACK : WHITE))
        {
            return {move,true};
        }
        
    }
    return {0,false};
}
double minimax_alpha_beta_q(Board & b, Engine & e,bool is_white, double alpha, double beta , int depth )
{
    PlayerColor color = b.data.player_to_play;
    bool we_are_player = (color == (is_white ? WHITE : BLACK));
    if(depth == 0) return evaluate(b,feature_map,weight_map);
    if(b.get_legal_moves().size() == 0){
        if(b.in_check()) return we_are_player ? -checkmate_score : checkmate_score;
        else return 0;
    }
    if(depth < 3)
    {
        double min_max_score = we_are_player ? -1e18 : 1e18;
        unordered_set<U16> moves = b.get_legal_moves();
        if(!e.search) return min_max_score;
        for(auto move : moves)
        {
            if(!e.search) return min_max_score;
            Board* bcopy = b.copy();
            bcopy->do_move(move);
            double score = minimax_alpha_beta_q(b,e,is_white,alpha,beta, depth-1);
            delete bcopy;
            if(we_are_player ? score > min_max_score : score < min_max_score) min_max_score = score;
            if(we_are_player) alpha = max(alpha, min_max_score);
            else beta = min(beta, min_max_score);
        }
        return min_max_score;
    }
    else{
        int cnt = 0;
        double min_max_score = we_are_player ? -1e18 : 1e18;
        unordered_set<U16> moves = b.get_legal_moves();

        if(!e.search) return min_max_score;
        for(auto move : moves)
        {
            if(!e.search) return min_max_score;
            if(cnt++ > 2) break;
            Board* bcopy = b.copy();
            bcopy->do_move(move);
            double score = minimax_alpha_beta_q(b,e,is_white,alpha,beta, depth-1);
            delete bcopy;
            if(we_are_player ? score > min_max_score : score < min_max_score) min_max_score = score;
            if(we_are_player) alpha = max(alpha, min_max_score);
            else beta = min(beta, min_max_score);
        }
        return min_max_score;
    }
    
}
double quaiscence_eval(Board& b, Engine& e,bool is_white,unordered_map<string,double>& weight_map,  int depth = QUAISCENCE_DEPTH)
{
    PlayerColor our_color = (is_white ? WHITE : BLACK);
    // if capture no capture possible at b stage
    if(debugmode){
        cout << "quiescence eval called at depth " << depth << endl;
    }
    if(depth == 0) {
        bool inchek = b.in_check();
        if(debugmode)
        {
            cout << "quiescence ended at depth " << depth << endl;
        }
        if(inchek) return our_color == b.data.player_to_play ? -checkmate_score : checkmate_score; // even after
                                                                                 // depths we are still in check
                                                                                 // we can't search any further
                                                                                 // possibly stalemate
        else return evaluate(b,feature_map,weight_map);
    }
    // if we are the player to play then we would straightaway return the evaluation
    if(b.data.player_to_play == (is_white ? WHITE : BLACK)) return evaluate(b,feature_map,weight_map);
    bool inchek = b.in_check();
    if(inchek && (b.get_legal_moves()).size()==0) {
        if(debugmode) cout << "quiescence ended at depth with checkmate " << depth << endl;
        return our_color == b.data.player_to_play ? -checkmate_score : checkmate_score;
    }
    if(inchek)
    {
        if(debugmode) cout << "quiescence called minimax_q" << depth << endl;
        return minimax_alpha_beta_q(b,e,is_white,-1e18,1e18,depth-1);
    } 
    auto [move, capture] = capture_possible(b);
    if(!capture) {
        if(debugmode) cout << "quiescence ended at depth with normal eval" << depth << endl;
        return evaluate(b,feature_map,weight_map);
    }
    else 
    {
        Board *bcopy = b.copy();
        bcopy->do_move(move);
        double cost = quaiscence_eval(*bcopy,e, depth-1,weight_map,QUAISCENCE_DEPTH);
        delete bcopy;
        if(debugmode) cout << "quiescence ended at depth after traversing " << depth << endl;
        return cost;
    }
};

pair<U16, double> q_iterative_alpha_beta(Board &b, Engine &e)
{
    auto start_time = std::chrono::system_clock::now().time_since_epoch().count(); //b is the start of the process.
    int cur_depth = min_depth; //defined as a global variable above.
    pair<U16,double> best_move = {0,0};
    pair<U16,double> actual_best_move = {0,0};
    auto end_time = std::chrono::system_clock::now().time_since_epoch().count();
    double time_since_start = 0;
    int counter = 0;
    vector<U16> actual_path;
    while(time_since_start + 0.01 < time_per_move && e.search && counter < max_depth)
    {
        vector<U16> path;
        best_move = queiscence_alpha_beta_minimax(b,e,cur_depth,path,-1e9,1e9); //gets the best move till the best_depth it can.
        if(e.search == false) break;
        counter++;
        if(best_move.first == 0)
        {
            cout << "actual best move being sent as 0 bruh" << endl;
            break; //do we really need to break here, even if we found a checkmate?.
        }
        else
        {
            actual_path = path;
            actual_best_move = best_move;
            e.best_move = actual_best_move.first; //setting the move.
            cur_depth++;
            update_weights(b, actual_best_move.second, counter); //, best_move.first); //updates the weights. ONLY IF update_weights is set to 1. //Updating at each depth value for now instead of just the end.
            if(actual_best_move.second >= checkmate_score) //we found a checkmate, so we can just return b.
            {
                cout << "WE FOUND A FORCED CHECKMATE at depth " << counter << endl;
                break; //we break at b point and do not go further, since we have found a forced checkmate
            }
            end_time = std::chrono::system_clock::now().time_since_epoch().count();
            time_since_start = (end_time - start_time)/1e9; //in seconds.
        }
    }
    if(debugmode)
    {
        cout << "went to depth " << counter << " at time " << time_since_start << "cost: " << actual_best_move.second;
        cout << ", path is: ";
        for(auto move : actual_path)
        {
            cout << move_to_str(move) << " <- ";
        }
        cout << "\n";
    } 
    return actual_best_move; //we keep increasing the depth as far as we can.
}
pair<U16, double> iterative_alpha_beta(Board &b, Engine &e)
{
    auto start_time = std::chrono::system_clock::now().time_since_epoch().count(); //b is the start of the process.
    int cur_depth = min_depth; //defined as a global variable above.
    pair<U16,double> best_move = {0,0};
    pair<U16,double> actual_best_move = {0,0};
    auto end_time = std::chrono::system_clock::now().time_since_epoch().count();
    double time_since_start = 0;
    int counter = 0;
    vector<U16> actual_path;
    vector<U16> best_moves(max_depth+1);
    while(time_since_start + 0.01 < time_per_move && e.search && counter < max_depth)
    {
        vector<U16> path;
        best_move = alpha_beta_minimax(b,e,cur_depth,path, actual_path, true); //gets the best move till the best_depth it can.
        best_moves[cur_depth] = best_move.first;
        if(e.search == false) break;
        counter++;
        if(best_move.first == 0)
        {
            cout << "actual best move being sent as 0 bruh" << endl;
            break; //do we really need to break here, even if we found a checkmate?.
        }
        else
        {
            actual_path = path;
            actual_best_move = best_move;
            e.best_move = actual_best_move.first; //setting the move.

//            cout << move_to_str(e.best_move) << " Set\n";
            cur_depth++;
            update_weights(b, actual_best_move.second, counter); //, best_move.first); //updates the weights. ONLY IF update_weights is set to 1. //Updating at each depth value for now instead of just the end.
            if(actual_best_move.second >= checkmate_score) //we found a checkmate, so we can just return b.
            {
                cout << "WE FOUND A FORCED CHECKMATE at depth " << counter << endl;
                break; //we break at b point and do not go further, since we have found a forced checkmate
            }
            if(actual_best_move.second <= -checkmate_score)
            {
                cout << "WE WERE FORCED A CHECKMATE at depth " << counter << endl;
                break;
            }
            end_time = std::chrono::system_clock::now().time_since_epoch().count();
            time_since_start = (end_time - start_time)/1e9; //in seconds.
        }
    }
    if(debugmode)
    {
        cout << "went to depth " << counter << " at time " << time_since_start << "cost: " << actual_best_move.second;
        cout << ", path is: ";
        for(auto move : actual_path)
        {
            cout << move_to_str(move) << " <- ";
        }
        cout << "\n";
    } 
    return actual_best_move; //we keep increasing the depth as far as we can.
}
void Engine::find_best_move(const Board& b) 
{
    move_number++;
    //fill(stats_alpha_beta.begin(),stats_alpha_beta.end(),0); //clears the stats vector.
    our_color = (b.data.player_to_play); //gets our color.
    if(search == false) //search flag.
    {
        this->best_move = 0;
        return;
    }
    auto moveset = b.get_legal_moves();
    /*for(auto move : moveset)
    {
        cout << move_to_str(move) << " ";
    }
    cout << endl;
    string human_move;  cin >> human_move;
    this->best_move = str_to_move(human_move);
    while(moveset.count(this->best_move) == 0)
    {
        cout << "invalid move, try again" << endl;
        cin >> human_move;
        this->best_move = str_to_move(human_move);
    }
    return; */
    if (moveset.size() == 0) {
        this->best_move = 0;
    }
    else if(moveset.size() == 1)
    {
        this->best_move = *moveset.begin();
    }
    else
    {
        auto start_time = chrono::high_resolution_clock::now();
        Board* board_copy = b.copy();
        auto [m, cost] = (qmode ? q_iterative_alpha_beta(*board_copy, *this) : iterative_alpha_beta(*board_copy,*this)); //gets the best move till the best_depth it can.
        auto end_time =  chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        // b->best_move = m;
        if(debugmode >= 1)
        {
            for (auto moves : moveset)
            {
                cout << move_to_str(moves) << " ";
            }
            cout << endl;
            cout << move_number << "th best move: " << move_to_str(m) << " with cost " << cost << endl;
            cout << board_to_str(b.data.board_0) << endl;
            //cout << "score for white: " << fpoints(*board_copy,true) << "and black " << fpoints(*board_copy, false) << endl;
            cout << "alpha_beta_stats: " << endl;
            for(int i = 0; i < stats_alpha_beta.size(); i++)
            {
                cout << i << ": " << stats_alpha_beta[i] << endl;
            }
            cout << "legal moves are:" << endl;
            for(auto move : moveset)
            {
                cout << move_to_str(move) << " ";
            }
        } 
        delete board_copy;
        //the best move is already being set in iterative_alpha_beta function now.
    }    
}
double evaluate(Board& b,  std::unordered_map<string,feature>& features,std::unordered_map<string,double> &w_map)
{
    double score = 0; //for now, we assume symmetric weight distribution accross enemy and us.
    for(auto f : feature_map)
    {
        double w = w_map.at(f.first); bool we_are_white = (our_color == WHITE);
        if(w == 0 || w <= 0.001) continue;
        double inc = w * ((f.second)(b, we_are_white) - (f.second)(b, !we_are_white));
        if(isnan(inc))
        {
            cout << "b IS NAN " << f.first << endl;
            assert(false); /////CHANGE b
        }
        score += inc;
    }
    //may later add some activation function, or a method to update the weights.
    return score;
}
pair<U16,double> queiscence_alpha_beta_minimax(Board &b, Engine &e, int cur_depth,vector<U16> &path,double alpha,double beta)
{
    // if(!e.search) return {0,1e18}; //time is up. Just return whatever we have right now.
    bool is_white = (our_color == WHITE);
    if(cur_depth == 0) return {0,e.should_update ?  quaiscence_eval(b,e,is_white,weight_map,QUAISCENCE_DEPTH) : quaiscence_eval(b,e,is_white,untrained_weight_map,QUAISCENCE_DEPTH)}; //returns the move and the cost.
    auto moveset = b.get_legal_moves();
    if(moveset.size() == 0) 
    {
        //return {0, e.should_update ? evaluate(b,feature_map,weight_map) : evaluate(b,feature_map, untrained_weight_map)};
        if(b.in_check())
        {
            //then it is a checkmate, and we return a very high or a very low cost.
            if(b.data.player_to_play == our_color) return {0,-checkmate_score};
            else return {0,checkmate_score}; 
        }
        else
        {
            //else I think we should return 0. based on the game state.
            return {0,0}; //Assuming 0 holds for a draw, so if we are in a disadvantageous position we will try to do a draw. and If we are in an advantageous position we will try to ensure that we do not draw the game.
        }
    }
    bool we_are_max = (b.data.player_to_play == our_color);
    double min_max_cost = we_are_max ? -1e18 : 1e18;
    U16 best_move = 1;
    int move_number = 0; int total_moves = moveset.size();
    vector<pair<double,U16>> cur_moves;
    bool we_are_white = (our_color == WHITE);
    for(auto move : moveset)
    {   
        if(!e.search) return {best_move,min_max_cost}; //time is up. Just return whatever we have right now.

        b.do_move(move); //b switches the player to play
        cur_moves.push_back({fpoints(b,we_are_white) - fpoints(b, !we_are_white) , move}); //the difference in points is a lightweight evaluation function to help prune more nodes.
        private_code::p_flip_player(&b);
        private_code::p_undo_last_move(move,&b);//undos the move.
        //cout << string(2*cur_depth, ' ') << cur_depth << " trying " << move_to_str(move) << "\n";
        //cout << string(2*cur_depth, ' ') << cur_depth << " costis " << cost << "\n";
    }
    if(we_are_max)
    {
        sort(cur_moves.begin(),cur_moves.end(),greater<pair<double,U16>>()); //sorting greatest first incase we are max_node.
    }
    else
    {
        sort(cur_moves.begin(),cur_moves.end()); //sorting minimum first incase we are min_node.
    }
    for(auto [order_cost, move] : cur_moves)
    {
        if(!e.search) return {best_move,min_max_cost};
        move_number++; 
        vector<U16> best_path;
        Board* bcopy = b.copy();
        bcopy->do_move(move); //b one also switches the player to play
        auto [m,cost] = queiscence_alpha_beta_minimax(*bcopy,e,cur_depth - 1,best_path,alpha,beta);
        delete bcopy;

        if(!e.search) return {0,min_max_cost}; //time is up.
        if( we_are_max ? (cost > min_max_cost): cost < min_max_cost)
        {
            min_max_cost = cost;
            best_move = move;
            path = best_path;
            //b is the best path that we have.
            if(cost >= checkmate_score) //we found a checkmate, so we can just return b.
            {
                if(cost >= 1e18)
                {
                    cout << "1e18 cost at move " << move_to_str(move) << endl;
                    assert(false); //error detection.
                }
                path.push_back(best_move);
                if(we_are_max)
                {
                    //cout << "Found a forced checkmate" << endl;
                    return {best_move,checkmate_score};
                }
                //return {best_move,min_max_cost};
            }
        } 
        if(we_are_max) 
        { 
            alpha = max(alpha,min_max_cost);
        }
        else 
        { 
            beta = min(beta,min_max_cost);
        }
        if(alpha >= beta) 
        {
            while(stats_alpha_beta.size() < cur_depth+1) stats_alpha_beta.push_back(0);
            stats_alpha_beta[cur_depth] += total_moves - 1 - move_number;
            path.push_back(best_move);
            return {best_move,min_max_cost};
        }
    }
    path.push_back(best_move);
    return {best_move,min_max_cost}; 
}
pair<U16,double> alpha_beta_minimax(Board &b, Engine &e, int cur_depth,vector<U16> &path,const vector<U16> &prev_best_moves,bool on_prev_best_path,double alpha, double beta)
{
    // if(!e.search) return {0,1e18}; //time is up. Just return whatever we have right now.
    if(cur_depth == 0) return {0,e.should_update ?  evaluate(b,feature_map,weight_map) : evaluate(b,feature_map, untrained_weight_map)}; //returns the move and the cost.
    auto moveset = b.get_legal_moves();
    if(moveset.size() == 0) 
    {
        //return {0, e.should_update ? evaluate(b,feature_map,weight_map) : evaluate(b,feature_map, untrained_weight_map)};
        if(b.in_check())
        {
            //then it is a checkmate, and we return a very high or a very low cost.
            if(b.data.player_to_play == our_color) return {0,-checkmate_score};
            else return {0,checkmate_score}; 
        }
        else
        {
            //else I think we should return 0. based on the game state.
            return {0,0}; //Assuming 0 holds for a draw, so if we are in a disadvantageous position we will try to do a draw. and If we are in an advantageous position we will try to ensure that we do not draw the game.
        }
    }
    bool we_are_max = b.data.player_to_play == our_color;
    double min_max_cost = we_are_max ? -1e18 : 1e18;
    U16 best_move = 1;
    U16 previous_best_move = 0; //definitely not a legal move originally.
    int move_number = 0; int total_moves = moveset.size();
    vector<pair<double,U16>> cur_moves;
    bool we_are_white = (our_color == WHITE);
    //now we select the ideal move, if available.
    if(cur_depth >= 2 && on_prev_best_path)
    {
        previous_best_move = prev_best_moves[cur_depth - 2];
        assert(moveset.count(previous_best_move) > 0);
        moveset.erase(previous_best_move); //erasing the previous best move from the moveset.
    }
    for(auto move : moveset)
    {   
        if(!e.search) return {0, we_are_max ? -1e18 : 1e18}; //time is up. Just return whatever we have right now.
        b.do_move(move); //b switches the player to play
        cur_moves.push_back({fpoints(b,we_are_white) - fpoints(b, !we_are_white) , move}); //the difference in points is a lightweight evaluation function to help prune more nodes.
        (private_code::p_flip_player(&b)); //b switches back to us.
        private_code::p_undo_last_move(move,&b); //undos the move.
    }
    if(we_are_max)
    {
        sort(cur_moves.begin(),cur_moves.end()); //sorting minimum first incase we are max_node, as at the end we will iterate in reverse order.
    }
    else
    {
        sort(cur_moves.begin(),cur_moves.end(),greater<pair<double,U16>>()); //sorting greatest first incase we are min_node, as at the end we will iterate in reverse order.
    }
    
    if(cur_depth >= 2 && on_prev_best_path)
    {
        //now we insert our move at the end of our ordered list cur_moves, which we will iterate in reverse order.
        cur_moves.push_back({0, previous_best_move}); //now this move will be checked first.
    }

    //NOW we are iterating in reverse, hence changed the order of sorting in the upper loops as well
    for(auto move_pair = cur_moves.rbegin(); move_pair != cur_moves.rend(); move_pair++)
    {
        auto [order_cost, move] = *move_pair;
        if(!e.search) return {0, we_are_max ? -1e18 : 1e18};
        move_number++; 
        vector<U16> best_path;
        Board* bcopy = b.copy();
        bcopy->do_move(move); //b one also switches the player to play
        auto [m,cost] = alpha_beta_minimax(*bcopy,e,cur_depth - 1,best_path,prev_best_moves,(move == previous_best_move),alpha,beta);
        delete bcopy;
        if(!e.search) return {0,we_are_max ? -1e18 : 1e18}; //time is up.
        if( we_are_max ? (cost > min_max_cost): cost < min_max_cost)
        {
            min_max_cost = cost;
            best_move = move;
            path = best_path;
            //b is the best path that we have.
            if(cost >= checkmate_score) //we found a checkmate, so we can just return b.
            {
                if(cost >= 1e18)
                {
                    cout << "1e18 cost at move " << move_to_str(move) << endl;
                    assert(false); //error detection.
                }
                if(we_are_max)
                {
                    path.push_back(best_move);
                    //cout << "Found a forced checkmate" << endl;
                    return {best_move,checkmate_score};
                }
            }
            else if(cost <= -checkmate_score)
            {
                if(!we_are_max)
                {
                    path.push_back(best_move);
                    return {best_move,-checkmate_score};
                }
            }
        } 
        if(we_are_max) 
        { 
            alpha = max(alpha,min_max_cost);
        }
        else 
        { 
            beta = min(beta,min_max_cost);
        }
        if(alpha >= beta) 
        {
            while(stats_alpha_beta.size() < cur_depth+1) stats_alpha_beta.push_back(0);
            stats_alpha_beta[cur_depth] += total_moves - 1 - move_number;
            path.push_back(best_move);
            return {best_move,min_max_cost};
        }
    }
    path.push_back(best_move);
    return {best_move,min_max_cost}; 
}

void humanplay(const Board& b, Engine *e) 
{
    std::string move;
    std::cout << all_boards_to_str(b) << std::endl;
    std::cin >> move;
    while(move.length() != 4)
    {
        std::cout << "Invalid move, format: x1y1x2y2" << std::endl;
        std::cin >> move;
    }
    std::pair<U8, U8> from = {move[0] - 'a', move[1] - '1'};
    std::pair<U8,U8> to = {move[2] - 'a', move[3] - '1'};
    U16 execute = move(pos(from.first, from.second), pos(to.first, to.second));
    auto moveset = b.get_legal_moves();
    if (moveset.size() == 0) {
        e->best_move = 0;
    }
    else {
        if(moveset.find(execute) != moveset.end())
        {
            e->best_move = execute;
        }
        else
        {
            std::cout << "Invalid move" << std::endl;
            humanplay(b, e);
        }
    }
}
