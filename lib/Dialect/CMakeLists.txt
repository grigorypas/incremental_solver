add_mlir_dialect_library(MLIRModelFormulation
        ModelFormulationDialect.cpp
        ModelFormulationOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/plco/

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIROptLib
        )
