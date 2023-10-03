#pragma once

#include "board.hpp"
#include <atomic>

class Engine {

    public:
    std::atomic<U16> best_move;
    std::atomic<bool> search;
    int should_update;
    Engine();
    Engine(int should_update);
    virtual void find_best_move(const Board& b);
};
