import unittest
import numpy as np
from em import EMEstimator

class TestEMEstimator(unittest.TestCase):
    def setUp(self):
        self.word_occur = np.array([[1, 2, 3], [1, 2, 3]])
        self.estimator = EMEstimator()
    
    def test_init_params(self):
        print(self.estimator.init_params(self.word_occur))
if __name__ == '__main__':
    unittest.main()