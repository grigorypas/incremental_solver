#include "IncrementalSolver/Common/Types.h"
#include "IncrementalSolver/ExpressionGraph/interface.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace incremental_solver;
using namespace incremental_solver::expression_graph;

void init_variables(py::module &m) {
  py::enum_<ValueType>(m, "ValueType")
      .value("INTEGER", ValueType::INTEGER)
      .value("DOUBLE", ValueType::DOUBLE)
      .export_values();

  py::class_<ValueExpression>(m, "ValueExpression")
      .def_property_readonly("value_type", &ValueExpression::getValueType)
      .def_property_readonly("id", &ValueExpression::getId)
      .def("get_int_value", &ValueExpression::getIntValue)
      .def("get_double_value", &ValueExpression::getDoubleValue);

  py::class_<DecisionVariable, ValueExpression>(m, "DecisionVariable")
      .def("set_int_value", &DecisionVariable::setIntValue)
      .def("set_double_value", &DecisionVariable::setDoubleValue);
}
