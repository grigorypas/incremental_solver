import unittest

from incremental_solver.interface import Engine


class TestInterface(unittest.TestCase):
    def test_var_creation(self):
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

    def test_mul_div_add(self):
        engine = Engine()
        var1 = engine.create_integer_variable(5)
        var2 = engine.create_integer_variable(3)
        expr1 = engine(10 * var1)
        expr2 = engine(var1 * 5)
        expr3 = engine(var1 / 10)
        expr4 = engine(var1 + var2)
        expr5 = engine(var1 + 7)
        engine.compile()
        self.assertEqual(expr1.value, 50)
        self.assertEqual(expr2.value, 25)
        self.assertAlmostEqual(expr3.value, 0.5, 2)
        self.assertEqual(expr4.value, 8)
        self.assertEqual(expr5.value, 12)


if __name__ == '__main__':
    unittest.main()
