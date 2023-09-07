#include <algorithm>
#include <random>
#include <iostream>
#include <map>
#include<unordered_map>
#include <climits>
#include <map>
#include <string>
#include "board.hpp"
#include "engine.hpp"
// #include<fstream>
#include<chrono>
#define curtime = std::chrono::system_clock::now().time_since_epoch().count()
using namespace std;
using feature = double (*)(const Board& b); 
void humanplay(const Board& b, Engine *e);
int debugmode = 1;
bool white = true;
std::unordered_map<string,double> weight_map = {{"enemyPoints",-3},{"ourPoints",1}};
pair<U16, double> minimax(Board& b, Engine &e, int depth, bool is_min);
double ourPoints(const Board& b);
double enemyPoints(const Board& b);
double our_pieces_under_attack(const Board& b);
double enemies_pieces_under_attack(const Board& b);

void Engine::find_best_move(const Board& b) 
{
    white = (b.data.player_to_play == WHITE); //gets our color.
    cout << "our color is " << (white ? "white" : "black") << endl;
    // pick a random move
    // humanplay(b, this);
    // return;
    b.copy(); 
    if(search == false)
    {
        this->best_move = 0;
        return;
    }
    auto moveset = b.get_legal_moves();
    if(debugmode)
        cout << "total moves possible: " << moveset.size() << endl;
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
        auto [m, val] = minimax(*board_copy, *this, 2, false);
        delete board_copy;
        auto end_time =  chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        cout << "starting points: " << ourPoints(b);
        cout << "starting enemy points: " << enemyPoints(b) << endl;
        cout << "time taken to find move " << move_to_str(m) << ": " << duration.count() << endl;
        cout << "value of move: " << val << endl;
        this->best_move = m;
    }    
}

//need to store parameters.
//our points
//opponents points
//our king position
//the total pawns near queening. possibly non-linear
//number of legal moves remaining of the king
//number of pieces under attack
//number of pawns remaining.

double ourPoints(const Board& b)
{
    string board = board_to_str(b.data.board_0);
    double score = 0;
    if(white)
    {
        for(size_t i = 0; i < board.length(); i++)
        {
            if(board[i] == 'P') score += 1;
            else if(board[i] == 'R') score += 3;
            else if(board[i] == 'B') score += 5;
        }
    }
    else
    {
        for(size_t i = 0; i < board.size(); i++)
        {
            if(board[i] == 'p') score += 1;
            else if(board[i] == 'r') score += 3;
            else if(board[i] == 'b') score += 5;
        }
    }
    return score;
}

double enemyPoints(const Board& b)
{
    string board = board_to_str(b.data.board_0); 
    double score = 0;
    if(!white)
    {
        for(size_t i = 0; i < board.size(); i++)
        {
            if(board[i] == 'P') score += 1;
            else if(board[i] == 'R') score += 3;
            else if(board[i] == 'B') score += 5;
        }
    }
    else
    {
        for(size_t i = 0; i < board.size(); i++)
        {
            if(board[i] == 'p') score += 1;
            else if(board[i] == 'r') score += 3;
            else if(board[i] == 'b') score += 5;
        }
    }
    return score;
}


std::unordered_map<string,feature> feature_map = {{"enemyPoints",enemyPoints},{"ourPoints",ourPoints}};

inline double evaluate(const Board& b,  std::unordered_map<string,feature>& features,std::unordered_map<string,double> &weight_map)
{
    double score = 0;
    for(auto f : feature_map) score += weight_map.at(f.first) * (f.second)(b);
    //may later add some activation function, or a method to update the weights.
    return score;
}

void undo_move(Board& b, U16 move)
{
    b._flip_player();
    b._undo_last_move(move);
}
void minimax_with_path(Board& b, Engine &e, int depth, bool is_min, vector<U16>& path, double& cost)
{

    if(depth == 0) {
        cost = evaluate(b,feature_map,weight_map);
        return;
    }
    auto moveset = b.get_legal_moves();
    vector<U16> best_possible_moves;
    if(is_min)
    {
        double min_val = 1e18;
        U16 best_move = 1<<15; //dummy value
        for(auto cur_move : moveset)
        {
            b.do_move(cur_move);
            double val;
            minimax_with_path(b,e,depth-1,false,path,val);
            undo_move(b, cur_move); //flips the player.
            cout << "mini :: " << move_to_str(cur_move) << ": "<< val << endl;
            if(val < min_val)
            {
                best_possible_moves.clear();
                min_val = val;
                best_possible_moves.push_back(cur_move);
            }
            else if(val == min_val)
            {
                best_possible_moves.push_back(cur_move);
            }
        }
        if(best_possible_moves.size() == 0)
        {
            cout << "didn't find a best move" << endl;
            path.push_back(0);
            return;
        }
        if(best_possible_moves.size() == 1)
        {
            best_move = best_possible_moves[0];
            if(best_move == 0) cout << "best move is 1<<15, how??????" << endl;
            path.push_back(best_move);
            cost = min_val;
            return;
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<int> dis(0, best_possible_moves.size() - 1);
        best_move = best_possible_moves[dis(g)];
        path.push_back(best_move);
        cost = min_val;
        return;
    }
    double max_val = -1e18;
    U16 best_move = 0;
    for(auto cur_move : moveset)
    {
        b.do_move(cur_move);
        auto [m, val] = minimax(b, e, depth - 1, true);
        undo_move(b, cur_move); //flips the player. 
        cout << "max value for " << move_to_str(cur_move) << ": "<< val << endl;
        if(val > max_val)
        {
            best_possible_moves.clear();
            max_val = val;
            best_possible_moves.push_back(cur_move);
        }
        else if(val == max_val)
        {
            best_possible_moves.push_back(cur_move);
        }
    }
    if(best_possible_moves.size() == 0)
    {
        cout << "max didn't find a best move" << endl;
        return;
    }
    if(best_possible_moves.size() == 1)
    {
        best_move = best_possible_moves[0];
        if(best_move == 0) cout << "max best move is 0, how??????" << endl;
        path.push_back(best_move);
        cost = min_val;
        return;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dis(0, best_possible_moves.size() - 1);
    best_move = best_possible_moves[dis(g)];
    path.push_back(best_move);
    cost = min_val;
    return;
}


std::pair<U16,double> minimax(Board& b, Engine &e, int depth, bool is_min)
{
    if(depth == 0) return  {0, evaluate(b, feature_map, weight_map)};
    auto moveset = b.get_legal_moves();
    vector<U16> best_possible_moves;
    if(is_min)
    {
        double min_val = 1e18;
        U16 best_move = 1<<15; //dummy value
        for(auto cur_move : moveset)
        {
            b.do_move(cur_move);
            auto [m, val] = minimax(b, e, depth - 1, false);
            undo_move(b, cur_move); //flips the player.
            cout << "mini :: " << move_to_str(cur_move) << ": "<< val << endl;
            if(val < min_val)
            {
                best_possible_moves.clear();
                min_val = val;
                best_possible_moves.push_back(cur_move);
            }
            else if(val == min_val)
            {
                best_possible_moves.push_back(cur_move);
            }
        }
        if(best_possible_moves.size() == 0)
        {
            cout << "didn't find a best move" << endl;
            return {1<<15, min_val};
        }
        if(best_possible_moves.size() == 1)
        {
            best_move = best_possible_moves[0];
            if(best_move == 1<<15) cout << "best move is 1<<15, how??????" << endl;
            return {best_move, min_val};
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<int> dis(0, best_possible_moves.size() - 1);
        best_move = best_possible_moves[dis(g)];
        return {best_move, min_val};
    }
    double max_val = -1e18;
    U16 best_move = 0;
    for(auto cur_move : moveset)
    {
        b.do_move(cur_move);
        auto [m, val] = minimax(b, e, depth - 1, true);
        undo_move(b, cur_move); //flips the player. 
        cout << "max value for " << move_to_str(cur_move) << ": "<< val << endl;
        if(val > max_val)
        {
            best_possible_moves.clear();
            max_val = val;
            best_possible_moves.push_back(cur_move);
        }
        else if(val == max_val)
        {
            best_possible_moves.push_back(cur_move);
        }
    }
    if(best_possible_moves.size() == 0)
    {
        cout << "max didn't find a best move" << endl;
        return {0, max_val};
    }
    if(best_possible_moves.size() == 1)
    {
        best_move = best_possible_moves[0];
        if(best_move == 0) cout << "max best move is 0, how??????" << endl;
        return {best_move, max_val};
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dis(0, best_possible_moves.size() - 1);
    best_move = best_possible_moves[dis(g)];
    return {best_move, max_val};
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
