import sys
sys.path.append('..')

import unittest
import os
import pandas as pd
from instrumental_data_processor import abstracts

class TestSignal(unittest.TestCase):
    
    def test_from_csv(self):
        abstracts.Signal.from_csv('tests/data/table_1.csv')

class TestSignal1D(unittest.TestCase):
    
    def test_from_csv(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_1_formatted.csv')
        self.assertEqual(signal.get_axis_name(), "column 1 (unit)")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "column 2 (unit)")
        self.assertEqual(signal.get_value_unit(), None)
        signal = abstracts.Signal1D.from_csv('tests/data/table_1_formatted.csv', detect_axis_name_and_unit=True, detect_value_name_and_unit=True)
        self.assertEqual(signal.get_axis_name(), "column 1")
        self.assertEqual(signal.get_axis_unit(), "unit")
        self.assertEqual(signal.get_value_name(), "column 2")
        self.assertEqual(signal.get_value_unit(), "unit")
    
    def test_get_axis(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
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
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv', detect_axis_name_and_unit=False, detect_value_name_and_unit=False)
        self.assertEqual(signal.get_values_between(1, 2).iloc[0], 2)
        
    def test_preview(self):
        signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
        # 断言以下代码运行时会出现 TypeError
        with self.assertRaises(TypeError):
            signal.preview(export_path="tests/out/table_2_preview_with_Signal1D") # No curve will be observed because the plot_at method is not implemented in the abstract class
class TestNumericSignal1D(unittest.TestCase):
    
    def test_get_peak_between(self):
        signal = abstracts.NumericSignal1D.from_csv('tests/data/table_2.csv', detect_axis_name_and_unit=False, detect_value_name_and_unit=False)
        self.assertEqual(signal.get_peak_between(1, 2), (1.1, 2))
        
    def test_preview(self):
        signal = abstracts.NumericSignal1D.from_csv(
            'tests/data/UV_Vis_1.csv'
            )
        self.assertEqual(signal.get_axis_name(), "Wavelength (nm)")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "1mm Absorbance")
        self.assertEqual(signal.get_value_unit(), None)
        signal.set_axis_name("Wavelength")
        signal.set_axis_unit("nm")
        signal.set_value_name("1 mm Absorbance")
        signal.set_value_unit(None)
        self.assertEqual(signal.get_axis_name(), "Wavelength")
        self.assertEqual(signal.get_axis_unit(), "nm")
        self.assertEqual(signal.get_value_name(), "1 mm Absorbance")
        self.assertEqual(signal.get_value_unit(), None)
        signal.preview(export_path="tests/out/UV_Vis_1_preview_with_NumericSignal1D")
        signal = abstracts.NumericSignal1D.from_csv(
            'tests/data/chromatography_UV_1.csv'
        )
        self.assertEqual(signal.get_axis_name(), "ml")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "mAU")
        self.assertEqual(signal.get_value_unit(), None)
        signal.set_axis_name("Volume")
        signal.set_axis_unit("ml")
        signal.set_value_name("280 nm")
        signal.set_value_unit("mAU")
        signal.preview(export_path="tests/out/chromatography_UV_1_preview_with_NumericSignal1D")
class TestFractionSignal(unittest.TestCase):

    def test_preview(self):
        fraction_signal = abstracts.FractionSignal.from_csv(
            'tests/data/fraction_1.csv',
            name=None,
            axis_unit="ml"
        )
        fraction_signal.set_axis_name("Volume")
        fraction_signal.preview(rotation=90, text_shift=(0, 0.05), export_path="tests/out/fraction_1_preview_with_FractionSignal.png")
        
if __name__ == '__main__':
    unittest.main()