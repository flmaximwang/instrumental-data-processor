from instrumental_data_analyzer.abstracts.signal_1d import ContinuousSignal1D
from ..abstracts.signal_1d import SegmentedContinuousSignal1D
from ..abstracts.signal_1d_collection import Signal1DCollection, ContinuousSignal1DCollection
from ..abstracts import DescAnno, ContDescAnno
import pandas as pd
import matplotlib.pyplot as plt

class Voltammegram(SegmentedContinuousSignal1D):
    
    def __init__(
        self, data: pd.DataFrame, name: str = "Voltammegram"
    ):
        '''
        A voltammetry signal should looks like\n
        Sequence,Potential (V),Current (A),Segment\n
        0,0.2,0.2,1\n
        1,0.22,0.3,1\n
        ...\n
        10,0.4,0.4,1\n
        11,0.38,0.38,2\n
        ...\n
        '''
        super().__init__(
            data,
            name,
            axis_name = "Potential",
            axis_unit = "V",
            value_name = "Current",
            value_unit = "A",
        )
        
    def get_current_limit(self):
        return self.get_axis_limit()
    
    def get_voltage_limit(self):
        return self.get_value_limit()

class VoltammegramCollection(ContinuousSignal1DCollection):
    
    def __init__(
        self,
        signals: list[Voltammegram] = [],
        name: str = "Default Voltammegram Collection",
        main_signal_name: str = None,
        visible_signal_names: list[str] = None,
        display_mode: str = None,
        figsize = None,
        axis_description = ContDescAnno(
            "Potential", "V", 
            tick_number=6, margin=(0.1, 0.1),
            digits=2
        ),
        value_description = ContDescAnno(
            "Current", "A",
            tick_number=6, margin=(0.1, 0.1),
            digits=1
        ),
    ):
        super().__init__(
            signals=signals,
            name=name,
            main_signal_name=main_signal_name,
            visible_signal_names=visible_signal_names,
            display_mode=display_mode,
            figsize=figsize,
            axis_description=axis_description,
            value_description=value_description
        )
        self.set_default_real_value_limit()
    
    def __getitem__(self, key: str):
        res: Voltammegram = super().__getitem__(key)
        return res