import unittest
import main

class TestSolver(unittest.TestCase):
#py -m unittest test_solver.py og py test_solver.py when __main__ is made
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_add(self):
        self.assertEqual(main.add(10, 5), 15)
        self.assertEqual(main.add(-1, 1), 0)
        self.assertEqual(main.add(-1, -1), -2)
    
    def test_divide(self):
        with self.assertRaises(ValueError):
            main.divide(10, 0)

if __name__ == '__main__':
    unittest.main()