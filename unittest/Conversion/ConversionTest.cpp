#include "IncrementalSolver/Conversion/ConvertToExpressionGraph.h"
#include "IncrementalSolver/ExpressionGraph/interface.h"
#include "IncrementalSolver/ModelBuilder/ModelFormulationBuilder.h"

#include <gtest/gtest.h>
#include <unordered_map>

using namespace incremental_solver::expression_graph;
using namespace incremental_solver;

template <typename T>
std::unordered_map<int64_t, T> convertToMap(const std::vector<T> input) {
  std::unordered_map<int64_t, T> result;
  for (const auto &var : input) {
    result.emplace(var.getId(), var);
  }
  return result;
}

TEST(Conversion, TrackEqualInput) {
  ModelFormulationBuilder modelBuilder;
  modelBuilder.addIntegerVarDecl(10, 1);
  modelBuilder.makeIntegerVarTracked(1);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(*modelBuilder.getModule(), vars, expressions);
  ASSERT_EQ(vars.size(), 1);
  ASSERT_EQ(expressions.size(), 1);
  auto &var = vars[0];
  auto &expr = expressions[0];
  ASSERT_EQ(expr.getIntValue(), 10);
  var.setIntValue(15);
  ASSERT_EQ(expr.getIntValue(), 15);
}

TEST(Conversion, SumInt) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addIntegerVarDecl(10, 1);
  auto v2 = modelBuilder.addIntegerVarDecl(15, 2);
  modelBuilder.emitIntegerSum({v1, v2}, 3);
  modelBuilder.makeIntegerVarTracked(3);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(*modelBuilder.getModule(), vars, expressions);
  auto varMap = convertToMap(vars);
  ASSERT_EQ(expressions.size(), 1);
  auto &expr = expressions[0];
  ASSERT_EQ(expr.getIntValue(), 25);
  // varMap[1].setIntValue(20);
  // ASSERT_EQ(expr.getIntValue(), 30);
}
