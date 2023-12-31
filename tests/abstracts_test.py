import unittest
import os
import pandas as pd
import instrumental_data_processor.abstracts as abstracts

class TestSignal(unittest.TestCase):
    
    def test_from_csv(self):
        abstracts.Signal.from_csv('tests/data/table_1.csv')

class TestSignal1D(unittest.TestCase):
    
    def test_from_csv(self):
        abstracts.Signal1D.from_csv('tests/data/table_1_formatted.csv')
    
    def test_get_axis(self):
        signal = abstracts.abstracts.Signal1D.from_csv('tests/data/table_2.csv')
        self.assertEqual(signal.get_axis().iloc[0], 1)
    
    def test_get_values(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
        self.assertEqual(signal.get_values().iloc[0], 2)
    
    def test_get_axis_between(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
        print("")
        print(signal.get_axis_between(1, 2))
        self.assertEqual(signal.get_axis_between(1, 2).iloc[0], 1.1)
    
    def test_get_values_between(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
        self.assertEqual(signal.get_values_between(1, 2).iloc[0], 2)
        
class TestNumericSignal1D(unittest.TestCase):
    
    def test_get_peak_between(self):
        signal = abstracts.NumericSignal1D.from_csv('tests/data/table_2.csv')
        self.assertEqual(signal.get_peak_between(1, 2), (1.1, 2))
        
class TestFractionSignal(unittest.TestCase):

    def test_preview(self):
        fraction_signal = abstracts.FractionSignal.from_csv('tests/data/fraction.csv', "ml")
        fraction_signal.set_axis_type("Volume")
        fraction_signal.preview(rotation=90)
        
if __name__ == '__main__':
    unittest.main()