from ..abstracts import ContinuousSignal1D

class AbsorbSpec(ContinuousSignal1D):
    
    def get_intensity_at_wavelength(self, wavelength):
        return self.get_value_at_axis(wavelength)