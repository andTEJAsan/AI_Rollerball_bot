#pragma once

#include <vector>
#include <unordered_set>
#include <stack>

typedef uint8_t U8;
typedef uint16_t U16;

#define pos(x,y) (((y)<<3)|(x))
#define gety(p)  ((p)>>3)
#define getx(p)  ((p)&0x7)

#define move(p0, p1) (((p0)<<8)|(p1))
#define move_promo(p0, p1, pt) (((p0)<<8)|(p1)|(pt))
#define getp0(m)    (((m)>>8)&0x3f)
#define getpromo(m) ((m)&(PAWN_BISHOP|PAWN_ROOK))
#define getp1(m)    ((m)&0x3f)

#define DEAD pos(7,7)

enum PlayerColor {
    WHITE=(1<<6),
    BLACK=(1<<5)
};

enum PieceType {
    EMPTY  = 0,
    PAWN   = (1<<1),
    ROOK   = (1<<2),
    KING   = (1<<3),
    BISHOP = (1<<4)

};

enum Promotion {
    PAWN_BISHOP = (1<<6),
    PAWN_ROOK   = (1<<7)
};

struct BoardData {

    // DO NOT add any fields above this
    U8 b_rook_ws  = pos(2,5);
    U8 b_rook_bs  = pos(2,6);
    U8 b_king     = pos(3,5);
    U8 b_bishop   = pos(3,6);
    U8 b_pawn_ws  = pos(4,5);
    U8 b_pawn_bs  = pos(4,6);

    U8 w_rook_ws  = pos(4,1);
    U8 w_rook_bs  = pos(4,0);
    U8 w_king     = pos(3,1);
    U8 w_bishop   = pos(3,0);
    U8 w_pawn_ws  = pos(2,1);
    U8 w_pawn_bs  = pos(2,0);
    
    U8 board_0[64];
    U8 board_90[64];
    U8 board_180[64];
    U8 board_270[64];

    PlayerColor player_to_play = WHITE;
    U8 last_killed_piece = 0;
    int last_killed_piece_idx = -1;

};

struct Board {

    BoardData data;

    Board();

    std::unordered_set<U16> get_legal_moves() const;
    bool in_check() const;
    Board* copy() const;
    void do_move(U16 move);

    private:
    std::unordered_set<U16> _get_pseudolegal_moves() const;
    std::unordered_set<U16> _get_pseudolegal_moves_for_piece(U8 piece_pos) const;
    void _flip_player();
    void _do_move(U16 move);
    bool _under_threat(U8 piece_pos) const;
    void _undo_last_move(U16 move);
    std::unordered_set<U16> _get_pseudolegal_moves_for_side(U8 color) const;
};

std::string move_to_str(U16 move);
U16 str_to_move(std::string move);
std::string board_to_str(const U8 *b);
std::string all_boards_to_str(const Board& b);
char piece_to_char(U8 piece);
