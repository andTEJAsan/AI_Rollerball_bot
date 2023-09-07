#include <algorithm>
#include <random>
#include <iostream>

#include "board.hpp"
#include "engine.hpp"

#ifdef move
#undef move
#define MOVE_WAS_DEFINED
#endif

#ifdef pos
#undef pos
#define POS_WAS_DEFINED
#endif

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#ifdef MOVE_WAS_DEFINED
#define move
#undef MOVE_WAS_DEFINED
#endif

#ifdef POS_WAS_DEFINED
#define pos
#undef POS_WAS_DEFINED
#endif

namespace py = pybind11;


void Engine::find_best_move(const Board& b) {
    // py::scoped_interpreter guard{};  // Start the Python interpreter
    py::scoped_interpreter guard{};  // Start the Python interpreter

    std::cout << "Starting Best Move" << std::endl;
    std::cout << "Loading Module" << std::endl;
    
    std::cout << "Module almost Loaded" << std::endl;
    py::module my_module;
    try{
        my_module = py::module::import("engine");
    }catch(const std::exception& e){
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    std::cout << "Module Loaded" << std::endl;
    py::object find_best_move_func = my_module.attr("find_best_move");
    std::cout << "In find_best_move" << std::endl;
    this->best_move = find_best_move_func(b).cast<int>();
    std::cout << "Best Move Found" << std::endl;
}