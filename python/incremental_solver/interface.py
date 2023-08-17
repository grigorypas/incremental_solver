import sys
from typing import Any, Optional, Dict, Tuple, Union, List
from enum import Enum
from abc import ABC, abstractmethod
from isbindings import ModelBuilder

NumberType = Union[int, float]
VarDataType = Union["TmpExpression", "Variable"]


class ValueType(Enum):
    INTEGER = 1
    CONTINUOUS = 2

    @property
    def is_integer(self) -> bool:
        return self == ValueType.INTEGER


class TmpExpression(ABC):
    def __init__(self) -> None:
        self._index: Optional[int] = None
        self._value_type: Optional[ValueType] = None
        self._id: Optional[int] = None

    def _set_id(self, id: int) -> None:
        self._id = id

    def _get_or_build(self, engine: "Engine") -> Tuple[int, ValueType]:
        if self._index is not None:
            return self._index, self._value_type
        self._index, self._value_type = self._build(engine)
        return self._index, self._value_type

    def _throw_if_modified(self) -> None:
        if self._index is not None:
            raise ValueError(
                "Cannot modify in-place. The expression has already been used")

    def _convert_to_index_and_type(
            self,
            var: VarDataType,
            engine: "Engine"
    ) -> Tuple[int, ValueType]:
        if isinstance(var, Variable):
            return engine._vars[var.id][1], var.value_type
        if isinstance(var, TmpExpression):
            return var._get_or_build(engine)
        raise ValueError("Unsupported type")

    @abstractmethod
    def _build(self, engine: "Engine") -> Tuple[int, ValueType]:
        pass


class LinearTmpExpression(TmpExpression):
    def __init__(self, bias: NumberType = 0) -> None:
        super().__init__()
        self._bias = bias
        self._vars: List[VarDataType] = []
        self._coefficients: List[NumberType] = []

    @staticmethod
    def create(
        vars: List[VarDataType],
        coefficients: List[NumberType],
        bias: NumberType = 0
    ) -> "LinearTmpExpression":
        if len(vars) != len(coefficients):
            raise ValueError(
                "Number of variables does not match the number of coefficients")
        expr = LinearTmpExpression(bias)
        expr._vars = list(vars)
        expr._coefficients = list(coefficients)
        return expr

    def _build(self, engine: "Engine") -> Tuple[int, ValueType]:
        if len(self._vars) == 0:
            return self._emit_const(self._bias)
        value_type = ValueType.INTEGER
        if isinstance(self._bias, float):
            value_type = ValueType.CONTINUOUS
        var_indeces = []
        for v, c in zip(self._vars, self._coefficients):
            ind, cur_value_type = self._convert_to_index_and_type(v, engine)
            ind, cur_value_type = self._emit_multiply(
                ind, c, cur_value_type, engine)
            if cur_value_type == ValueType.CONTINUOUS:
                value_type = ValueType.CONTINUOUS
            var_indeces.append(ind)
        if self._bias != 0:
            var_indeces.append(self._emit_const(self._bias))
        if value_type == ValueType.INTEGER:
            ind = engine._model_builder.emit_integer_sum(var_indeces, self._id)
        else:
            ind = engine._model_builder.emit_double_sum(var_indeces, self._id)
        return ind, value_type

    def _emit_multiply(
            self,
            ind: int,
            c: NumberType,
            var_value_type: ValueType,
            engine: "Engine"
    ) -> Tuple[int, ValueType]:
        if c == 1:
            return ind, var_value_type
        if var_value_type == ValueType.CONTINUOUS or isinstance(c, float):
            return engine._model_builder.emit_double_multiply(ind, c, None), ValueType.CONTINUOUS
        return engine._model_builder.emit_integer_multiply(ind, c, None), ValueType.INTEGER

    def _emit_const(
            self,
            c: NumberType,
            engine: "Engine",
            id: Optional[int]
    ) -> Tuple[int, ValueType]:
        if isinstance(c, float):
            return engine._model_builder.emit_double_const_assignment(c, id), ValueType.CONTINUOUS
        return engine._model_builder.emit_integer_const_assignment(c, id), ValueType.INTEGER


class Variable(object):
    __slots__ = ("_id", "_is_integer", "_var")

    def __init__(self, id: int, is_integer: bool) -> None:
        self._id = id
        self._is_integer = is_integer
        self._var = None

    @property
    def id(self) -> int:
        return self._id

    @property
    def is_integer(self) -> bool:
        return self._is_integer

    @property
    def is_continuous(self) -> bool:
        return not self._is_integer

    @property
    def value_type(self) -> ValueType:
        if self._is_integer:
            return ValueType.INTEGER
        return ValueType.CONTINUOUS

    @property
    def value(self) -> float:
        if self._var is None:
            raise ValueError("Compile model before using")
        if self._is_integer:
            return self._var.get_int_value()
        return self._var.get_double_value()

    @value.setter
    def value(self, val: float) -> None:
        raise ValueError(
            "Setting values is not supported on derived variables")


class DecisionVariable(Variable):
    __slots__ = ("_lb", "_ub")

    def __init__(
            self,
            id: int,
            is_integer: bool,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None
    ) -> None:
        super().__init__(id, is_integer)
        self._lb = lower_bound
        self._ub = upper_bound

    @Variable.value.setter
    def value(self, val: float) -> None:
        if self._var is None:
            raise ValueError("Compile model before using")
        if self._is_integer and isinstance(val, float):
            raise ValueError(
                "Cannot set float value on integer decision variable")
        if self._lb is not None and val < self._lb:
            raise ValueError("Provided value is less than lower bound")
        if self._ub is not None and val > self._ub:
            raise ValueError("Provided value is greater than upper bound")
        if self._is_integer:
            self._var.set_int_value(val)
        else:
            self._var.set_double_value(val)


class Engine:
    def __init__(self) -> None:
        self._model_builder = ModelBuilder()
        self._vars: Dict[int, Tuple[Variable, int]] = {}
        self._compiled: bool = False
        self._cur_id = -1

    def create_integer_variable(
            self,
            value: int,
            lower_bound: Optional[int] = None,
            upper_bound: Optional[int] = None
    ) -> DecisionVariable:
        self._throw_if_comiled()
        id = self._create_id()
        if lower_bound is not None and value < lower_bound:
            raise ValueError("Value is less than lower bound")
        if upper_bound is not None and value > upper_bound:
            raise ValueError("Value is greater than upper bound")
        var = DecisionVariable(id, True, lower_bound, upper_bound)
        ind = self._model_builder.add_integer_var_decl(
            value, id, lower_bound, upper_bound)
        self._vars[id] = (var, ind)
        return var

    def create_binary_variable(self, value: bool) -> DecisionVariable:
        if value:
            return self.create_integer_variable(1, 0, 1)
        return self.create_integer_variable(0, 0, 1)

    def create_continous_variable(
            self,
            id: int,
            value: float,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None
    ) -> DecisionVariable:
        self._throw_if_comiled()
        id = self._create_id()
        if lower_bound is not None and value < lower_bound:
            raise ValueError("Value is less than lower bound")
        if upper_bound is not None and value > upper_bound:
            raise ValueError("Value is greater than upper bound")
        var = DecisionVariable(id, False, lower_bound, upper_bound)
        ind = self._model_builder.add_double_var_decl(
            value, id, lower_bound, upper_bound)
        self._vars[id] = (var, ind)
        return var

    def compile(self) -> None:
        if self._compiled:
            return
        self._compiled = True
        # adding what variables must be tracked
        decision_vars_map = {}
        derived_vars_map = {}
        for id, var_ind in self._vars.items():
            if isinstance(var_ind[0], DecisionVariable):
                decision_vars_map[id] = var_ind[0]
            elif sys.getrefcount(var_ind[0]) > 2:
                self._model_builder.mark_as_tracked(id)
                derived_vars_map[id] = var_ind[0]
        del self._vars
        self._model_builder.compile()
        decision_vars = self._model_builder.get_decision_variables()
        for var in decision_vars:
            decision_vars_map[var.id]._var = var
        tracked_vars = self._model_builder.get_tracked_expressions()
        for var in tracked_vars:
            derived_vars_map[var.id]._var = var
        self._model_builder = None

    def __call__(self, expr: Union[NumberType, VarDataType]) -> Variable:
        self._throw_if_comiled()
        if isinstance(expr, Variable):
            return expr
        if isinstance(expr, NumberType):
            expr = LinearTmpExpression(expr)
        id = self._create_id()
        expr._set_id(id)
        ind, value_type = expr._get_or_build(self)
        var = Variable(id, value_type.is_integer)
        self._vars[id] = (var, ind)
        return var

    def _throw_if_comiled(self) -> None:
        if self._compiled:
            raise ValueError("The model cannot be modified after compilation")

    def _create_id(self) -> int:
        self._cur_id += 1
        return self._cur_id
