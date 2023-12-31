import typing
import pandas as pd
import matplotlib.pyplot as plt
from .signal import Signal

chromatographic_units_map_types = {
    "min": "Time",
    "mL": "Volume",
    "mAU": "Absorbance",
}

spectroscopic_units_map_types = {
    "nm": "Wavelength",
    "": "Absorbance",
}

class Signal1D(Signal):
    def __init__(self, data: pd.DataFrame, name: str):
        super().__init__(data, name)
        self.axis_type = "undefined"
        self.axis_unit = None
        self.value_type = "undefined"
        self.value_unit = None

    def get_axis(self):
        return self.data.iloc[:, 0]

    def get_axis_between(self, start, end):
        """
        获取 axis 在 start 到 end 之间的值
        """
        axis_filter = (self.data.iloc[:, 0] > start) & (self.data.iloc[:, 0] < end)
        return self.get_axis()[axis_filter]

    def get_values(self):
        return self.data.iloc[:, 1]

    def get_values_between(self, start, end):
        """
        获取 axis 在 start 到 end 之间时 value 的值
        """
        axis_filter = (self.data.iloc[:, 0] > start) & (self.data.iloc[:, 0] < end)
        return self.get_values()[axis_filter]
    
    def get_axis_type(self):
        return self.axis_type
    
    def get_axis_unit(self):
        return self.axis_unit
    
    def get_value_type(self):
        return self.value_type
    
    def get_value_unit(self):
        return self.value_unit
    
    def plot_at(self, ax: plt.Axes, label, **kwargs) -> plt.Axes:
        pass

class NumericSignal1D(Signal1D):


    def __mul__(self, factor):
        """
        Multiply the values of the signal by the given factor.
        This method enables the * operator to be used for multiplication.
        """
        values = self.get_values()
        values *= factor
    
    def get_peak_between(self, start, end) -> typing.Tuple[float, float]:
        """
        寻找当 axis 位于 start 与 end 之间时, value 的最大值以及对应的 axis, 返回一个 (axis, value) 的元组
        """
        peak_idx = self.get_values_between(start, end).idxmax()
        peak_axis = self.get_axis_between(start, end)[peak_idx]
        peak_value = self.get_values_between(start, end)[peak_idx]
        return (peak_axis, peak_value)

    def plot_at(self, ax: plt.Axes, label, **kwargs):
        """
        Plot the signal at the given ax and return an artist handle.
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        handle, = ax.plot(self.get_axis(), self.get_values(), label, **kwargs)
        return handle

    def plot_peak_at(
        self, ax: plt.Axes, start, end, type="vline", text_shift=(0, 0), **kwargs
    ):
        """
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        peak_axis, peak_value = self.get_peak_between(start, end)
        if type == "vline":
            self._plot_peak_at_with_vline(
                ax, peak_axis, peak_value, text_shift, **kwargs
            )
        elif type == "annotation":
            self._plot_peak_at_with_annotation(ax, peak_axis, peak_value, **kwargs)

    def _plot_peak_at_with_vline(
        self, ax: plt.Axes, peak_axis, peak_value, text_shift, **kwargs
    ):
        linestyle = kwargs.pop("linestyle", "dashed")
        ax.vlines(peak_axis, 0, peak_value, linestyles=linestyle, **kwargs)
        ax.annotate(
            f"{peak_axis:.2f} {self.axis_unit}",
            xy=(peak_axis, peak_value),
            xytext=(peak_axis+text_shift[0], peak_value+text_shift[1]),
            **kwargs,
        )

    def _plot_peak_at_with_annotation(
        self, ax: plt.Axes, peak_axis, peak_value, *args, **kwargs
    ):
        ax.annotate(
            f"{peak_axis:.2f} {self.axis_unit}",
            xy=(peak_axis, peak_value),
            xytext=(peak_axis, peak_value * 1.1),
            arrowprops=dict(
                arrowstyle="-",
                color=kwargs.get("color", "black"),
                linewidth=kwargs.get("linewidth", 1),
            ),
            **kwargs,
        )
        
    def preview(self, **kwargs):
        fig, ax = plt.subplots(1, 1)
        self.plot_at(ax, label=self.name)
        if self.axis_unit:
            ax.set_xlabel(f"{self.axis_type} ({self.axis_unit})")
        else:
            ax.set_xlabel(f"{self.axis_type}")
        if self.value_unit:
            ax.set_ylabel(f"{self.value_type} ({self.value_unit})")
        else:
            ax.set_ylabel(f"{self.value_type}")
        ax.set_xlim(self.get_axis().min(), self.get_axis().max())
        ax.set_ylim(0, 1)
        ax.legend()
        plt.show()

class AnnotationSignal1D(Signal1D):
    
    def plot_at(self, ax: plt.Axes, label, text_shift=(0, 0), mark_height=0.5, **kwargs):
        """
        Plot the signal at the given ax and return an artist handle.
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        kwargs_for_annotate = kwargs.copy()
        kwargs_for_Line2D = kwargs.copy()
        
        kwargs_for_Line2D.pop("rotation", None)
        ax.vlines(self.get_axis(), 0, mark_height, **kwargs_for_Line2D)
        for (axis, value) in zip(self.get_axis(), self.get_values()):
            ax.annotate(
                f"{value}",
                xy=(axis, mark_height),
                xytext=(axis+text_shift[0], 0.5+text_shift[1]),
                arrowprops=dict(
                    arrowstyle="-",
                    color=kwargs.get("color", "black"),
                    linewidth=kwargs.get("linewidth", 1),
                ),
                **kwargs_for_annotate,
            )
            
        handle = plt.Line2D([0], [0], label=label, **kwargs_for_Line2D) # Generate a virtural handle for legend
        return handle
    
    def preview(self, export_path=None, **kwargs):
        fig, ax = plt.subplots(1, 1)
        handles = []
        handles.append(self.plot_at(ax, label=self.name, **kwargs))
        if self.axis_unit:
            ax.set_xlabel(f"{self.axis_type} ({self.axis_unit})")
        else:
            ax.set_xlabel(f"{self.axis_type}")
        if self.value_unit:
            ax.set_ylabel(f"{self.value_type} ({self.value_unit})")
        else:
            ax.set_ylabel(f"{self.value_type}")
        ax.set_xlim(self.get_axis().min(), self.get_axis().max())
        ax.set_ylim(0, 1)
        ax.legend(handles=handles)
        if export_path:
            fig.savefig(export_path)
        else:
            plt.show()
        
class FractionSignal(AnnotationSignal1D):
    
    def __init__(self, data, name, axis_type):
        super().__init__(data, name)
        self.axis_type = axis_type
        self.axis_unit = None
        self.value_type = "Fraction"
        self.value_unit = None