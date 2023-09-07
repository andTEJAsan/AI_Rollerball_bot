#include <algorithm>
#include <random>
#include <iostream>

#include "board.hpp"
#include "engine.hpp"
void humanplay(const Board& b, Engine *e);


void Engine::find_best_move(const Board& b) {

    // pick a random move
    humanplay(b, this);
    return;
    if(search == false)
    {
        this->best_move = 0;
        return;
    }
    auto moveset = b.get_legal_moves();
    if (moveset.size() == 0) {
        this->best_move = 0;
    }
    else {
        std::vector<U16> moves;
        std::cout << all_boards_to_str(b) << std::endl;
        for (auto m : moveset) {
            std::cout << move_to_str(m) << " ";
        }
        std::cout << std::endl;
        std::sample(
            moveset.begin(),
            moveset.end(),
            std::back_inserter(moves),
            1,
            std::mt19937{std::random_device{}()}
        );
        this->best_move = moves[0];
    }
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
