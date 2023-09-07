#pragma once

#include "board.hpp"
#include <atomic>
#include <unordered_map>
#include<string>

class Engine {
    
    public:
    std::atomic<U16> best_move;
    std::atomic<bool> search;
    
    virtual void find_best_move(const Board& b);
};
