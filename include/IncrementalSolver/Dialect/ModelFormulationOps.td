#ifndef MODEL_FORMULATION_OPS
#define MODEL_FORMULATION_OPS

include "IncrementalSolver/Dialect/ModelFormulationDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Base PLCO operation definition.
//===----------------------------------------------------------------------===//

class IS_OpInOut<string mnemonic, list<Trait> traits = []> :
        Op<IS_ModelFormulationDialect, mnemonic, traits>;


class IS_Op<string mnemonic, Type ResultType, list<Trait> traits = []> :
        Op<IS_ModelFormulationDialect, mnemonic, traits # [Pure]>, Results<(outs ResultType:$result)>;

class IS_VariadicInputOp<string mnemonic, Type InputType, Type ResultType, Attr BoundsType, list<Trait> traits = []>:
        IS_Op<mnemonic, ResultType, traits>, 
        Arguments<(ins Variadic<InputType>:$inputVector, BoundsType:$lowerBound, BoundsType:$upperBound)>{
    let assemblyFormat = "attr-dict `(` $inputVector `)` `:` type($result)"; 
}

def IS_IntegerTrackedVarDecl: IS_OpInOut<"integerTrackedVar">{
    let summary = "Declare integer expression being tracked";
    let description = [{
        If an expression is tracked, it cannot be optimized away.
        Its value can be queried by user. 
    }];
    let arguments = (ins I64:$value, I64Attr:$id, I64Attr:$lowerBound, I64Attr:$upperBound);
    let assemblyFormat = "$value ` ` `[` `id` `=` $id `]` attr-dict";
}

def IS_DoubleTrackedVarDecl: IS_OpInOut<"doubleTrackedVar">{
    let summary = "Declare integer expression being tracked";
    let description = [{
        If an expression is tracked, it cannot be optimized away.
        Its value can be queried by user. 
    }];
    let arguments = (ins F64:$value, I64Attr:$id);
    let assemblyFormat = "$value ` ` `[` `id` `=` $id `]` attr-dict";
}

def IS_IntegerDecisionVarDecl: IS_OpInOut<"integerDecisionVar">{
    let summary = "Declare integer decision variable";
    let description = [{
        This operation declares integer decision variable. Decision 
        variables are input from the user. When declaring, the user
        has to provide initial value. Lower and upper bounds can
        be optionaly provided. The bounds are used at compile time
        as metadata for optimization purposes. 
    }];
    let arguments = (ins SI64Attr:$value, I64Attr:$id, SI64Attr:$lowerBound, SI64Attr:$upperBound);
    let results = (outs I64:$res); 
}

def IS_DoubleDecisionVarDecl: IS_OpInOut<"doubleDecisionVar">{
    let summary = "Declare continuous decision variable";
    let description = [{
        This operation declares continuous decision variable. Decision 
        variables are input from the user. When declaring, the user
        has to provide initial value. Lower and upper bounds can
        be optionaly provided. The bounds are used at compile time
        as metadata for optimization purposes. 
    }];
    let arguments = (ins F64Attr:$value, I64Attr:$id, F64Attr:$lowerBound, F64Attr:$upperBound);
} 

def IS_IntegerSumOp: IS_VariadicInputOp<"sumi", I64, I64, SI64Attr, [Commutative, SameOperandsAndResultType]>{
    let summary = "Sum of integer operands";
    let description = [{
        Sums up provided integer operands.
    }];
}

def IS_DoubleSumOp: IS_VariadicInputOp<"sumf", F64, F64, F64Attr, [Commutative, SameOperandsAndResultType]>{
    let summary = "Sum of continuous variables";
    let description = [{
        Sum of continous variables. 
    }];
}

def IS_IntegerComparisonOp: IS_Op<"cmpi", I64>{
    let summary = "Compares lhs with rhs";
    let description = [{
        Comparison operator. LHS and RHS have to be integers. 
        Types of comparison: equal, not equal, less than or equal and
        greater than or equal.
    }];
    let arguments = (ins IS_CmpPredicateAttr:$predicate, I64:$lhs, I64:$rhs);
}

def IS_DoubleComparisonOp: IS_Op<"cmpf", I64>{
    let summary = "Compares lhs with rhs";
    let description = [{
        Comparison operator. LHS and RHS have to be continuous variables. 
        Types of comparison: less than or equal and
        greater than or equal.
    }];
    let arguments = (ins IS_CmpPredicateAttr:$predicate, F64:$lhs, F64:$rhs);
}

def IS_AnyOp : IS_VariadicInputOp<"sumi", I64, I64, SI64Attr>{
    let summary = "Determines if at least one of the input binary values is True";
    let description = [{
        This operations takes a vector of binary variables and returns
        true if at least one of the values is true.
    }];
    
    let hasCanonicalizer = 1;   
}

#endif // MODEL_FORMULATION_OPS
