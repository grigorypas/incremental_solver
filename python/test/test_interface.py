import unittest

from incremental_solver.interface import Engine, LinearTmpExpression


class TestInterface(unittest.TestCase):
    def test_var_creation(self) -> None:
        engine = Engine()
        var1 = engine.create_integer_variable(5)
        var2 = engine.create_continuous_variable(0.5)
        engine.compile()
        self.assertEqual(var1.value, 5)
        self.assertEqual(var2.value, 0.5)
        var2.value = 1.2
        self.assertEqual(var2.value, 1.2)
        var1.value = 7
        self.assertEqual(var1.value, 7)

    def test_mul_div_add(self) -> None:
        engine = Engine()
        var1 = engine.create_integer_variable(5)
        var2 = engine.create_integer_variable(3)
        expr1 = engine(10 * var1)
        expr2 = engine(var1 * 5)
        expr3 = engine(var1 / 10)
        expr4 = engine(var1 + var2)
        expr5 = engine(var1 + 7)
        expr6 = engine(var1 - var2)
        expr7 = engine(12 - var2)
        engine.compile()
        self.assertEqual(expr1.value, 50)
        self.assertEqual(expr2.value, 25)
        self.assertAlmostEqual(expr3.value, 0.5, 2)
        self.assertEqual(expr4.value, 8)
        self.assertEqual(expr5.value, 12)
        self.assertEqual(expr6.value, 2)
        self.assertEqual(expr7.value, 9)

    def test_lin_expr_in_place_arith(self) -> None:
        engine = Engine()
        var1 = engine.create_integer_variable(5)
        var2 = engine.create_integer_variable(7)
        expr1 = LinearTmpExpression()
        expr1 += 3 * var1
        expr1 -= 2 * var2
        expr1 = engine(expr1)
        expr2 = LinearTmpExpression()
        expr2 += 4
        expr2 += var2
        expr2 *= 2
        expr2_tmp = expr2
        expr2 = engine(expr2)
        expr3 = LinearTmpExpression.create([var1, var2], [1, 1])
        expr3 /= 2
        expr3 = engine(expr3)
        expr4 = LinearTmpExpression()
        expr4 += 1
        expr4 += expr3
        expr4_tmp = expr4
        expr4 = engine(expr4)
        expr5 = LinearTmpExpression()
        expr5 += expr2_tmp
        expr5 += expr4_tmp
        expr5 = engine(expr5)
        engine.compile()
        self.assertEqual(expr1.value, 1)
        self.assertEqual(expr2.value, 22)
        self.assertAlmostEqual(expr3.value, 6.0, 2)
        self.assertAlmostEqual(expr4.value, 7.0, 2)
        self.assertAlmostEqual(expr5.value, 29)
        var1.value += 1
        self.assertEqual(expr1.value, 4)
        self.assertEqual(expr2.value, 22)
        self.assertAlmostEqual(expr3.value, 6.5, 2)
        self.assertAlmostEqual(expr4.value, 7.5, 2)
        self.assertAlmostEqual(expr5.value, 29.5)

    def test_lin_exp_add_sub_mul(self) -> None:
        engine = Engine()
        var1 = engine.create_integer_variable(5)
        var2 = engine.create_continuous_variable(2.5)
        expr1 = var1 + 2 * var2
        expr2 = 2 * var1
        expr3 = engine(expr1 - expr2)
        expr4 = engine(expr1 + expr2)
        expr5 = engine((expr1 + expr2) * 0.25)
        expr6 = engine((- expr1) + (- var1))
        engine.compile()
        self.assertAlmostEqual(expr3.value, 0.0, 2)
        self.assertAlmostEqual(expr4.value, 20.0, 2)
        self.assertAlmostEqual(expr5.value, 5.0, 2)
        self.assertAlmostEqual(expr6.value, -15.0, 2)


if __name__ == '__main__':
    unittest.main()
