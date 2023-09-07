#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "board.hpp"

namespace py = pybind11;

PYBIND11_MODULE(board, m) {
    py::class_<Board>(m, "Board")
        .def("get_legal_moves", &Board::get_legal_moves)
        .def("in_check", &Board::in_check)
        .def("copy", &Board::copy)
        .def("do_move", &Board::do_move);
}
