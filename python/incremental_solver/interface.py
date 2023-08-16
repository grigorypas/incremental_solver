import sys
from typing import Optional, Dict, Tuple
from isbindings import ModelBuilder


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

    def create_integer_variable(
            self,
            id: int,
            value: int,
            lower_bound: Optional[int] = None,
            upper_bound: Optional[int] = None
    ) -> DecisionVariable:
        self._throw_if_comiled()
        if lower_bound is not None and value < lower_bound:
            raise ValueError("Value is less than lower bound")
        if upper_bound is not None and value > upper_bound:
            raise ValueError("Value is greater than upper bound")
        if id in self._vars:
            raise ValueError(f"Variable with id {id} already exists")
        var = DecisionVariable(id, True, lower_bound, upper_bound)
        ind = self._model_builder.add_integer_var_decl(
            value, id, lower_bound, upper_bound)
        self._vars[id] = (var, ind)
        return var

    def create_binary_variable(self, id: int, value: bool) -> DecisionVariable:
        if value:
            return self.create_integer_variable(id, 1, 0, 1)
        return self.create_integer_variable(id, 0, 0, 1)

    def create_continous_variable(
            self,
            id: int,
            value: float,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None
    ) -> DecisionVariable:
        self._throw_if_comiled()
        if id in self._vars:
            raise ValueError(f"Variable with id {id} already exists")
        if lower_bound is not None and value < lower_bound:
            raise ValueError("Value is less than lower bound")
        if upper_bound is not None and value > upper_bound:
            raise ValueError("Value is greater than upper bound")
        var = DecisionVariable(id, False, lower_bound, upper_bound)
        ind = self._model_builder.add_double_var_decl(
            value, id, lower_bound, upper_bound)
        self._vars[id] = (var, ind)
        return var

    def _throw_if_comiled(self) -> None:
        if self._compiled:
            raise ValueError("The model cannot be modified after compilation")

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
