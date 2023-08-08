#include "IncrementalSolver/ExpressionGraph/interface.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_variables(py::module &m);
void init_model_builder(py::module &m);

PYBIND11_MODULE(_incremental_solver, m) {
  m.doc() = "Module of building model formulation";

  init_variables(m);
  init_model_builder(m);
}