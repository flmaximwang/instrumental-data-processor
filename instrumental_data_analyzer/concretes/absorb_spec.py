from ..abstracts import ContinuousSignal1D, ContinuousSignal1DCollection, ContDescAnno

class AbsorbSpec(ContinuousSignal1D):
    
    def get_intensity_at_wavelength(self, wavelength):
        return self.get_value_at_axis(wavelength)

class AbsorbSpecCollection(ContinuousSignal1DCollection):
    
    def __init__(
        self,
        signals: list[AbsorbSpec] = [],
        name: str = "Default AbsorbSpecCollection",
        main_signal_name: str = None,
        visible_signal_names: list[str] = None,
        display_mode: str = None,
        figsize = None,
        axis_description: ContDescAnno = None,
        value_description = None
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