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
        chemstation_chromatography.set_figsize((15, 5))
        chemstation_chromatography.align_signal_axes((0, 30))
        chemstation_chromatography.set_main_signal("280")
        chemstation_chromatography.set_visible_signals("280", "214")
        chemstation_chromatography.set_display_mode(0)
        chemstation_chromatography.plot()

class TestUnicornChrom(unittest.TestCase):
    
    def test_read_and_export(self):
        unicorn_chroms = instruments.read_uni_chroms_from_raw_export('tests/data/unicorn_chrom_1.txt')
        unicorn_chrom = unicorn_chroms[0]
        unicorn_chrom.set_figsize((25, 5))
        fig, axes = unicorn_chrom.plot(fontsize=5)
        fig.savefig("/Users/maxim/Documents/VSCode/instrumental-data-processer/tests/out/unicorn_chrom_1_preview_with_UnicornChrom.png")
        unicorn_chrom.set_display_mode(1)
        unicorn_chrom.align_signal_axes((0, 30))
        fig, axes = unicorn_chrom.plot(fontsize=10, axis_shift=0.04)
        a: instruments.UnicornContinuousSignal1D = unicorn_chrom["UV"]
        a.plot_peak_at(axes[0], 10, 15)
        fig.savefig("/Users/maxim/Documents/VSCode/instrumental-data-processer/tests/out/unicorn_chrom_1_preview_with_UnicornChrom_all_values.png")

if __name__ == '__main__':
    unittest.main()
    