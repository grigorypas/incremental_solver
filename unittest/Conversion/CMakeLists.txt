add_gtest_ut(
	NAME TestConversion
	SRCS ConversionTest.cpp
	LIBS 
    ExpressionGraph
    modelConversion
    ModelFormulationBuilder
)