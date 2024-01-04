import sys
sys.path.append('..')

import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
from instrumental_data_processor import instruments

class TestChemStationChromatographyNumericSignal(unittest.TestCase):
    
    def test_read_and_export(self):
        chemstation_chromatography_numeric_signal = instruments.ChemStationChromatographyNumericSignal.from_raw_export(
            'tests/data/chemstation_chromatography_1/280.CSV'
        )
        chemstation_chromatography_numeric_signal.set_value_name("280 nm")
        with self.assertRaises(Exception):
            chemstation_chromatography_numeric_signal.export("tests/out/chromatography_UV_1_export_with_ChemStationChromatographyNumericSignal.csv")
        chemstation_chromatography_numeric_signal.export("tests/out/chromatography_UV_1_export_with_ChemStationChromatographyNumericSignal.csv", mode="replace")

class TestChemStationChromatography(unittest.TestCase):
    
    def test_read_and_export(self):
        chemstation_chromatography = instruments.ChemStationChromatography.from_raw_directory(
            'tests/data/chemstation_chromatography_1'
        )
        with self.assertRaises(Exception):
            chemstation_chromatography.export("tests/out/chemstation_chromatography_1")
        chemstation_chromatography.export("tests/out/chemstation_chromatography_1", mode="replace")

    def test_plot(self):
        chemstation_chromatography = instruments.ChemStationChromatography.from_raw_directory(
            'tests/data/chemstation_chromatography_1'
        )
        print(chemstation_chromatography.main_signal_name)
        print(chemstation_chromatography.visible_signal_names)
        chemstation_chromatography.set_figsize((15, 5))
        print(chemstation_chromatography.figsize)
        chemstation_chromatography.align_signal_axes((0, 30))
        chemstation_chromatography.set_main_signal("280")
        chemstation_chromatography.set_visible_signals("280", "214")
        chemstation_chromatography.set_display_mode(0)
        chemstation_chromatography.plot()

if __name__ == '__main__':
    unittest.main()
    