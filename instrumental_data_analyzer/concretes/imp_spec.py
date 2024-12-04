from ..abstracts import Signal, SignalCollection, DescAnno
# 如果有 impedance 包, 尝试导入以下的几个模块
try:
    import impedance.preprocessing as imp_pp
    import impedance.visualization as imp_v
except ImportError:
    pass
import pandas as pd
import matplotlib.pyplot as plt

class ImpedanceSpectrum(Signal):

    def __init__(self, data: pd.DataFrame, name: str = "Default Impedance Spectrum"):   
        '''
        An impedance spectrum signal should looks like\n
        Frequency (Hz),Z' (ohm),Z'' (ohm)\n
        0,0.2,0.2\n
        1,0.22,0.3\n
        ...\n
        10,0.4,0.4\n
        '''
        # 检查 data 的列名是否符合要求
        if not all([col in data.columns for col in ["Frequency (Hz)", "Z' (ohm)", "Z\" (ohm)"]]):
            raise ValueError("The column names of the data should be 'Frequency (Hz)', 'Z' (ohm)', 'Z\" (ohm))', while the current column names are " + str(data.columns))
        self.data = data
        self.name = name
        self.description_annotations = [DescAnno("Frequency", "Hz"), DescAnno("Z'", "\\Omega"), DescAnno("Z''", "\\Omega")]
        self.fmt = ".-"
        
    def plot_nyquist(self, ax: plt.Axes, color="C0"):
        frequencies = self.data["Frequency (Hz)"]
        Z = self.data["Z' (ohm)"] + self.data['Z" (ohm)'] * 1j
        imp_v.plot_nyquist(Z, ax=ax, fmt=self.fmt, label=self.get_name(), labelsize=14, units = self.get_description_annotations_by_index(1).unit, color=color)
        
    def plot_bode(self, axes: list[plt.Axes], color="C0"):
        frequencies = self.data["Frequency (Hz)"]
        Z = self.data["Z' (ohm)"] + self.data['Z" (ohm)'] * 1j
        imp_v.plot_bode(frequencies, Z, axes=axes, fmt=self.fmt, label=self.get_name(), labelsize=14, units = self.get_description_annotations_by_index(1).unit, color=color)

class ImpedanceSpectrumCollection(SignalCollection):
    
    def __init__(self, signals: list[ImpedanceSpectrum] = [], name="Default_ImpedanceSpectrum_collection") -> None:
        super().__init__(signals, name)
        self.colormap = "default"
        self.colormap_min = 0
        self.colormap_max = 1
        self.fmt = ".-"
        self.figsize = None
        
    def plot_nyquist(self, real_limit=None, imag_limit=None, scale = 1):
        '''
        - Scale: Only the width of figsize is valid when plotting nyquist due to the scale setting. Adjust y/x ratio with the scale parameter.
        '''
        if not self.figsize:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        for i, sig_name in enumerate(self):
            sig: ImpedanceSpectrum = self[sig_name]
            if isinstance(self.colormap, str):
                if self.colormap == "default":
                    sig.plot_nyquist(ax, color=f"C{i}")
                else:
                    sig.plot_nyquist(ax, color=plt.get_cmap(self.colormap)(i / len(self) * (self.colormap_max - self.colormap_min) + self.colormap_min))
            elif isinstance(self.colormap, list):
                sig.plot_nyquist(ax, color=self.colormap[i])
            else:
                sig.plot_nyquist(ax, color=f"C{i}")
                
        if real_limit:
            ax.set_xlim(real_limit)
        if imag_limit:
            ax.set_ylim(imag_limit)
        ax.legend()
        ax.set_title(self.get_name())
        
        return fig, ax
    
    def plot_bode(self, freq_limit=None, mag_limit=None, phase_limit=None, direction = "horizontal"):
        if direction == "horizontal":
            if not self.figsize:
                fig, axes = plt.subplots(1, 2)
            else:
                fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        elif direction == "vertical":
            if not self.figsize:
                fig, axes = plt.subplots(2, 1)
            else:
                fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        for i, sig_name in enumerate(self):
            sig: ImpedanceSpectrum = self[sig_name]
            if isinstance(self.colormap, str):
                if self.colormap == "default":
                    sig.plot_bode(axes, color=f"C{i}")
                else:
                    sig.plot_bode(axes, color=plt.get_cmap(self.colormap)(i / len(self) * (self.colormap_max - self.colormap_min) + self.colormap_min))
            elif isinstance(self.colormap, list):
                sig.plot_bode(axes, color=self.colormap[i])
            else:
                sig.plot_bode(axes, color=f"C{i}")
                
        if freq_limit:
            axes[0].set_xlim(freq_limit)
        if mag_limit:
            axes[0].set_ylim(mag_limit)
        if phase_limit:
            axes[1].set_ylim(phase_limit)
        axes[0].legend()
        axes[1].legend()
        axes[0].set_title(self.get_name())
        
        return fig, axes