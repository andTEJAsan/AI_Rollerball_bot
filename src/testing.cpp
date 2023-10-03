#include<map>
#include<cassert>
#include <fstream>
#include <iostream>
#include<thread>
#include<unistd.h>
#include "engine.cpp"
#include "board.hpp"
using namespace std;
U8 str_to_pos(string s)
{
    if(s=="dd") return pos(7,7);
    return (s[0] - 'a') + (s[1] - '1') * 8;
}
void create_board(Board& b){
    int tt = 0;
    U8* pieces = (U8*) &(b.data);
    while(tt < 12)
    {
        cout << "tt : " << tt  << "\n";
        string s;
        cin >> s;
        U8 pos = str_to_pos(s);
        pieces[tt] = pos;
        tt++;
    }
    for(int i = 0 ; i < 64; i++) b.data.board_0[i] = 0;
    b.data.board_0[b.data.b_rook_ws]  = BLACK | ROOK;
    b.data.board_0[b.data.b_rook_bs]  = BLACK | ROOK;
    b.data.board_0[b.data.b_king   ]  = BLACK | KING;
    b.data.board_0[b.data.b_bishop ]  = BLACK | BISHOP;
    b.data.board_0[b.data.b_pawn_ws]  = BLACK | PAWN;
    b.data.board_0[b.data.b_pawn_bs]  = BLACK | PAWN;
    b.data.board_0[b.data.w_rook_ws]  = WHITE | ROOK;
    b.data.board_0[b.data.w_rook_bs]  = WHITE | ROOK;
    b.data.board_0[b.data.w_king   ]  = WHITE | KING;
    b.data.board_0[b.data.w_bishop ]  = WHITE | BISHOP;
    b.data.board_0[b.data.w_pawn_ws]  = WHITE | PAWN;
    b.data.board_0[b.data.w_pawn_bs]  = WHITE | PAWN;
    assert(b.data.b_rook_ws==pieces[0]);   
    assert(b.data.b_rook_bs==pieces[1]);
    assert(b.data.b_king   ==pieces[2]);
    assert(b.data.b_bishop ==pieces[3]);
    assert(b.data.b_pawn_ws==pieces[4]);
    assert(b.data.b_pawn_bs==pieces[5]);
    assert(b.data.w_rook_ws==pieces[6]);
    assert(b.data.w_rook_bs==pieces[7]);
    assert(b.data.w_king   ==pieces[8]);
    assert(b.data.w_bishop ==pieces[9]);
    assert(b.data.w_pawn_ws==pieces[10]);
    assert(b.data.w_pawn_bs==pieces[11]);
    
    cout << board_to_str(b.data.board_0) << endl;
    cout << "Board created successfully\n";
}
void save_board(string output_filename)
{
    Board b;
    create_board(b);
    ofstream fout(output_filename);
    U8* pieces = (U8*) &(b.data);
    for(int i= 0; i < 12 ; i++)
    {
        fout << pieces[i] << "\n";
    }
}
void read_board(Board& b, string input_file)
{
    ifstream fin(input_file);
    U8* pieces = (U8*) &(b.data);
    for(int i = 0 ; i < 12 ; i++)
    {
        fin >> pieces[i];
    }
    for(int i = 0 ; i < 64; i++) b.data.board_0[i] = 0;
    b.data.board_0[b.data.b_rook_ws]  = BLACK | ROOK;
    b.data.board_0[b.data.b_rook_bs]  = BLACK | ROOK;
    b.data.board_0[b.data.b_king   ]  = BLACK | KING;
    b.data.board_0[b.data.b_bishop ]  = BLACK | BISHOP;
    b.data.board_0[b.data.b_pawn_ws]  = BLACK | PAWN;
    b.data.board_0[b.data.b_pawn_bs]  = BLACK | PAWN;
    b.data.board_0[b.data.w_rook_ws]  = WHITE | ROOK;
    b.data.board_0[b.data.w_rook_bs]  = WHITE | ROOK;
    b.data.board_0[b.data.w_king   ]  = WHITE | KING;
    b.data.board_0[b.data.w_bishop ]  = WHITE | BISHOP;
    b.data.board_0[b.data.w_pawn_ws]  = WHITE | PAWN;
    b.data.board_0[b.data.w_pawn_bs]  = WHITE | PAWN;
    assert(b.data.b_rook_ws==pieces[0]);   
    assert(b.data.b_rook_bs==pieces[1]);
    assert(b.data.b_king   ==pieces[2]);
    assert(b.data.b_bishop ==pieces[3]);
    assert(b.data.b_pawn_ws==pieces[4]);
    assert(b.data.b_pawn_bs==pieces[5]);
    assert(b.data.w_rook_ws==pieces[6]);
    assert(b.data.w_rook_bs==pieces[7]);
    assert(b.data.w_king   ==pieces[8]);
    assert(b.data.w_bishop ==pieces[9]);
    assert(b.data.w_pawn_ws==pieces[10]);
    assert(b.data.w_pawn_bs==pieces[11]);
    cout << board_to_str(b.data.board_0) << endl;
    cout << "Board read successfully\n";
}
void create_board_from_rep(Board&b,string ifs)
{

    ifstream fin(ifs);
    string color_to_move = "";
    getline(fin,color_to_move); //sets the color to move;
    if(color_to_move=="white") b.data.player_to_play = WHITE;
    else b.data.player_to_play = BLACK;
    U8* pieces = (U8*) &(b.data);
    map<char,stack<int>> m;
    m['r'].push(0);
    m['r'].push(1);
    m['k'].push(2);
    m['b'].push(3);
    m['p'].push(4);
    m['p'].push(5);
    m['R'].push(6);
    m['R'].push(7);
    m['K'].push(8);
    m['B'].push(9);
    m['P'].push(10);
    m['P'].push(11);
    // how to search in map with a given key
    // if(m.find("r") != m.end())
    for(int i = 0 ; i < 12; i++) pieces[i] = DEAD;
    for(int i = 6 ; i >= 0; i--)
    {
        string s;
        getline(fin,s);
        for(int j = 0; j < 7; j++)
        {
            char p = s[j];
            if(m.find(p)==m.end()) continue;
            if(m[p].empty()) continue;
            pieces[m[p].top()] = pos(j,i);
            m[p].pop();
        }

    }
    for(int i = 0 ; i < 64; i++) b.data.board_0[i] = 0;
    b.data.board_0[b.data.b_rook_ws]  = BLACK | ROOK;
    b.data.board_0[b.data.b_rook_bs]  = BLACK | ROOK;
    b.data.board_0[b.data.b_king   ]  = BLACK | KING;
    b.data.board_0[b.data.b_bishop ]  = BLACK | BISHOP;
    b.data.board_0[b.data.b_pawn_ws]  = BLACK | PAWN;
    b.data.board_0[b.data.b_pawn_bs]  = BLACK | PAWN;
    b.data.board_0[b.data.w_rook_ws]  = WHITE | ROOK;
    b.data.board_0[b.data.w_rook_bs]  = WHITE | ROOK;
    b.data.board_0[b.data.w_king   ]  = WHITE | KING;
    b.data.board_0[b.data.w_bishop ]  = WHITE | BISHOP;
    b.data.board_0[b.data.w_pawn_ws]  = WHITE | PAWN;
    b.data.board_0[b.data.w_pawn_bs]  = WHITE | PAWN;    
    cout << board_to_str(b.data.board_0) << endl;
}


int main()
{
    //Engine e; 
    Board b;
    U16 bestmove;
    create_board_from_rep(b,"input.txt");
    // cout << evaluate(b, feature_map, weight_map) << endl;
    // thread t1(&Engine::find_best_move, &e, b);
    // sleep(2);
    // bestmove = e.best_move;
    // e.search = false;
    // t1.join(); 
    cout <<  getx(b.data.b_rook_bs) << "," << gety(b.data.b_rook_bs) << endl;
    auto blackrookmoves = private_code::p_get_pseudolegal_moves_for_piece(b.data.b_rook_bs,&b); 
    cout << "black rook pseudolegalmoves: " << endl;
    for (auto m : blackrookmoves)
    {
        cout << move_to_str(m) << ", ";
    }
    cout << endl;
    auto playermoves = b.get_legal_moves(); 
    cout << ((b.data.player_to_play == WHITE) ? "white " : "black " )<< "moves: " << endl;
    for (auto m : playermoves)
    {
        cout << move_to_str(m) << ", ";
    }
    cout << endl;
    // cout << "white under threat: " << under_attack(b,true) << endl;
    // cout << "black under threat: " << under_attack(b,false) << endl;
    // cout << "best move" << move_to_str(bestmove) << endl;
    return 0;
}
