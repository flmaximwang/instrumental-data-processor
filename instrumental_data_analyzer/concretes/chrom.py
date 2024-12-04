from ..abstracts import ContinuousSignal1D, DiscreteSignal1D
from ..abstracts import Signal1DCollection

class ChromSig(ContinuousSignal1D):
    pass

class ChromLog(DiscreteSignal1D):
    pass

class Chrom(Signal1DCollection):
    
    def correct_conc(self, conc_delay):
        '''
        校正浓度信号, 因为浓度信号和实际的盐浓度有一定的延迟
        '''
        data = self["Conc B"].get_data()
        data.iloc[:, 0] += conc_delay