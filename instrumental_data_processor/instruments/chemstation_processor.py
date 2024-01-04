import os
import pandas as pd
from instrumental_data_processor.abstracts.signal_1d import ContinuousSignal1D, DiscreteSignal1D, FractionSignal, Signal1D
from instrumental_data_processor.abstracts.signal_1d_collection import Signal1DCollection
from instrumental_data_processor.utils import path_utils

class ChemStationChromatographyNumericSignal(ContinuousSignal1D):
    
    @staticmethod
    def from_raw_export(file_path: str):
        if not (file_path.endswith(".csv") or file_path.endswith(".CSV")):
            raise ValueError(f"Invalid file path {file_path}, should be a csv file")
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} is not a file")
        if os.path.basename(file_path) in ["Fraction.csv", "fraction.csv", "Fraction.CSV", "fraction.CSV"]:
            raise ValueError(f"File {file_path} is a fraction file, use ChemStationChromatographyFractionSignal.from_raw_export instead")
        signal_data = pd.read_csv(file_path, header=None, encoding="utf-16 LE", sep="\t")
        value_name = path_utils.get_name_from_path(file_path)
        signal = ChemStationChromatographyNumericSignal(
            data = signal_data,
            name = value_name,
            axis_name = "Time",
            axis_unit = "min",
            value_name = value_name,
            value_unit = "mAU"
        )
        return signal

class ChemStationChromatographyFractionSignal(FractionSignal):
    
    @staticmethod
    def from_raw_export(file_path: str):
        if not (file_path.endswith(".csv") or file_path.endswith(".CSV")):
            raise ValueError(f"Invalid file path {file_path}, should be a csv file")
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} is not a file")
        if not os.path.basename(file_path) in ["Fraction.csv", "fraction.csv", "Fraction.CSV", "fraction.CSV"]:
            raise ValueError(f"File {file_path} is a not fraction file, use ChemStationChromatographyNumericSignal.from_raw_export instead")
        fraction_data = pd.read_csv(file_path, encoding="utf-8", sep="\t")
        signal_data = pd.DataFrame({
            "Time (min)": [],
            "Fraction": []
        })
        for index, fraction in fraction_data.iterrows():
            row1 = pd.DataFrame({
                "Time (min)": [fraction["Start"]],
                "Fraction": [fraction["AFC Loc"]]
            })
            row2 = pd.DataFrame({
                "Time (min)": [fraction["End"]],
                "Fraction": "waste"
            })
            signal_data = pd.concat([signal_data, row1, row2], ignore_index=True)
        signal = ChemStationChromatographyFractionSignal(
            data = signal_data,
            name = "Fraction",
            axis_name = "Time",
            axis_unit = "min",
            value_name = "Fraction",
            value_unit = None
        )
        return signal

class ChemStationChromatography(Signal1DCollection):
    
    @staticmethod
    def from_raw_directory(directory, name=None):
        '''
        从一个包含 Chemstation 导出数据的目录中读取数据
        '''
        signals: list[Signal1D] = []
        file_name: str
        for file_name in os.listdir(directory):
            if file_name.endswith(".csv") or file_name.endswith(".CSV"):
                if file_name in ["Fraction.csv", "fraction.csv", "Fraction.CSV", "fraction.CSV"]:
                    signals.append(ChemStationChromatographyFractionSignal.from_raw_export(os.path.join(directory, file_name)))
                else:
                    signals.append(ChemStationChromatographyNumericSignal.from_raw_export(os.path.join(directory, file_name)))
        if name:
            chromatogram = ChemStationChromatography(signals, name=name)
        else:
            if directory.endswith("/"):
                directory = directory[:-1]
            chromatogram = ChemStationChromatography(signals, name=path_utils.get_name_from_path(directory, extension=False))
            
        return chromatogram
    
    def __init__(self, signals, name="Default ChemStation Chromatogram"):
        super().__init__(signals, name=name)