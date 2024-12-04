import typing
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .signal import * 
from ..utils import path_utils, transform_utils

chromatographic_units_map_types = {
    "min": "Time",
    "mL": "Volume",
    "ml": "Volume",
    "mAU": "Absorbance",
}

spectroscopic_units_map_types = {
    "nm": "Wavelength",
    "": "Absorbance",
}

units_map_types = [chromatographic_units_map_types, spectroscopic_units_map_types]

class Signal1D(Signal):
    """
    1D signals are signals with two descriptions.\n
    The first is called axis (which is continuous),\n
    the second is called value (which may be continuous or discrete).\n
    Value describes the property of every point on the axis, like the absorbance of a sample at a certain wavelength.
    """

    @classmethod
    def from_csv(
        cls,
        path,
        name=None,
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        **kwargs,
    ):
        if name is None:
            name = path_utils.get_name_from_path(path)
        data: pd.DataFrame = pd.read_csv(path, **kwargs)
        if detect_axis_name_and_unit:
            axis_name, axis_unit = cls.get_type_and_unit_from_header(data.columns[0])
        else:
            if not axis_name:
                axis_name = data.columns[0]
            axis_unit = axis_unit
        if detect_value_name_and_unit:
            value_name, value_unit = cls.get_type_and_unit_from_header(data.columns[1])
        else:
            if not value_name:
                value_name = data.columns[1]
            value_unit = value_unit
        return cls(
            data,
            name,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            update="to_data",
        )

    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        update="to_data",
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
        **kwargs
    ):
        super().__init__(
            data=data,
            name=name,
        )
        if detect_axis_name_and_unit:
            axis_name, axis_unit = self.get_type_and_unit_from_header(
                self.get_data().columns[0]
            )
        else:
            if not axis_name:
                axis_name = self.get_data().columns[0]
            axis_unit = axis_unit
        if detect_value_name_and_unit:
            value_name, value_unit = self.get_type_and_unit_from_header(
                self.get_data().columns[1]
            )
        else:
            if not value_name:
                value_name = self.get_data().columns[1]
            value_unit = value_unit
        self.description_annotations = [
            ContDescAnno(axis_name, axis_unit, (0, 0), 6),
            DescAnno(value_name, value_unit),
        ]
        if update == "from_data":
            self.update_type_and_unit_from_data(
                detect_axis_name_and_unit, detect_value_name_and_unit
            )
        elif update == "to_data":
            self.update_types_and_units_to_data()
        else:
            raise ValueError(
                f"Expected update to be one of ['from_data', 'to_data'], but got {update}"
            )

    def copy(self):
        copied_signal = type(self)(
            self.get_data().copy(),
            self.get_name(),
            self.get_axis_name(),
            self.get_axis_unit(),
            self.get_value_name(),
            self.get_value_unit(),
            "to_data",
            False,
            False,
        )

        return copied_signal

    def get_axis(self):
        return self.data.iloc[:, 0]

    def get_indices_between(self, start, end):
        indices = self.get_data().index[(self.get_data().iloc[:, 0] > start) & (self.get_data().iloc[:, 0] < end)]
        return indices
    
    def get_indices_for_current_axis_limit(self):
        return self.get_indices_between(*self.get_axis_limit())

    def get_axis_between(self, start, end):
        """
        获取 axis 在 start 到 end 之间的值
        """
        indices = self.get_indices_between(start, end)
        return self.get_axis()[indices]

    def set_values(self, values):
        old_values = self.get_values()
        old_values = values

    def get_values(self):
        return self.data.iloc[:, 1]

    def get_values_between(self, start, end):
        """
        获取 axis 在 start 到 end 之间时 value 的值
        """
        indices = self.get_indices_between(start, end)
        return self.get_values()[indices]

    def get_axis_name(self):
        return self.get_description_annotations_by_index(0).get_name()

    def set_axis_name(self, axis_name):
        self.get_description_annotations_by_index(0).set_name(axis_name)
        self.update_types_and_units_to_data()

    def get_axis_unit(self):
        return self.get_description_annotations_by_index(0).get_unit() 

    def set_axis_unit(self, axis_unit):
        self.get_description_annotations_by_index(0).set_unit(axis_unit)
        self.update_types_and_units_to_data()
        
    def get_axis_label(self):
        return self.get_description_annotations_by_index(0).get_label()
    
    def get_axis_digits(self):
        return self.get_description_annotations_by_index(0).get_digits()
    
    def set_axis_digits(self, digits):
        self.get_description_annotations_by_index(0).set_digits(digits)
        
    def get_value_digits(self):
        return self.get_description_annotations_by_index(1).get_digits()
    
    def set_value_digits(self, digits):
        self.get_description_annotations_by_index(1).set_digits(digits)

    def get_value_name(self):
        return self.get_description_annotations_by_index(1).get_name()

    def set_value_name(self, value_type):
        self.get_description_annotations_by_index(1).set_name(value_type)
        self.update_types_and_units_to_data()

    def get_value_unit(self):
        return self.get_description_annotations_by_index(1).get_unit()

    def set_value_unit(self, value_unit):
        self.get_description_annotations_by_index(1).set_unit(value_unit)
        self.update_types_and_units_to_data()

    def get_value_label(self):
        return self.get_description_annotations_by_index(1).get_label()
    
    def get_axis_limit(self):
        return self.get_limit(0)

    def set_axis_limit(self, axis_limit):
        self.set_limit(0, axis_limit)
        
    def get_axis_margin(self):
        return self.get_margin(0)
    
    def set_axis_margin(self, axis_margin):
        self.set_margin(0, axis_margin)
    
    def get_axis_ticks(self):
        return self.get_description_annotations_by_index(0).get_ticks()
    
    def get_axis_ticklabels(self, digits=1):
        return self.get_description_annotations_by_index(0).get_ticklabels()
    
    def get_value_ticks(self):
        return self.get_description_annotations_by_index(1).get_ticks()
    
    def get_value_tick_labels(self, digits=1):
        return self.get_description_annotations_by_index(1).get_ticklabels()
    
    def set_axis_tick_number(self, tick_number):
        self.get_description_annotations_by_index(0).set_tick_number(tick_number)
        
    def set_value_tick_number(self, tick_number):
        self.get_description_annotations_by_index(1).set_tick_number(tick_number)

    def update_types_and_units_to_data(self):
        old_columns = self.get_data().columns
        if self.get_value_unit() and self.get_axis_unit():
            new_columns = [
                f"{self.get_axis_name()} ({self.get_axis_unit()})",
                f"{self.get_value_name()} ({self.get_value_unit()})",
            ]
        elif self.get_value_unit():
            new_columns = [
                f"{self.get_axis_name()}",
                f"{self.get_value_name()} ({self.get_value_unit()})",
            ]
        elif self.get_axis_unit():
            new_columns = [
                f"{self.get_axis_name()} ({self.get_axis_unit()})",
                f"{self.get_value_name()}",
            ]
        else:
            new_columns = [f"{self.get_axis_name()}", f"{self.get_value_name()}"]
        self.get_data().rename(
            columns=dict(zip(old_columns, new_columns)), inplace=True
        )

    def update_type_and_unit_from_data(
        self, detect_axis_name_and_unit=True, detect_value_name_and_unit=True
    ):
        if detect_axis_name_and_unit:
            axis_name, axis_unit = self.get_type_and_unit_from_header(
                self.get_data().columns[0]
            )
        else:
            axis_name, axis_unit = self.get_data().columns[0], None
        if detect_value_name_and_unit:
            value_name, value_unit = self.get_type_and_unit_from_header(
                self.get_data().columns[1]
            )
        else:
            value_name, value_unit = self.get_data().columns[1], None
        self.set_axis_name(axis_name)
        self.set_axis_unit(axis_unit)
        self.set_value_name(value_name)
        self.set_value_unit(value_unit)

    def plot_at(self, ax: plt.Axes, label=None, **kwargs):
        '''
        The method to plot a signal should only be implemented when the form of signal has been well defined.
        Such a method should retrun a Line2D
        '''
        handle, = ax.plot([0], [0], label=label)
        return handle

    def preview(self, export_path=None, **kwargs):
        if type(self) is Signal1D:
            raise TypeError("An abstract Signal1D should not be previewed.")
        fig, ax = plt.subplots(1, 1)
        handle = self.plot_at(ax, **kwargs)
        ax.set_xlabel(self.get_axis_label())
        ax.set_ylabel(self.get_value_label())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(self.get_axis_ticks())
        ax.set_xticklabels(self.get_axis_ticklabels())
        ax.set_yticks(self.get_value_ticks())
        ax.set_yticklabels(self.get_value_tick_labels())
        ax.legend(handles=[handle])
        if export_path:
            fig.savefig(export_path)
        else:
            plt.show()
    
    def move_along_axis(self, distance):
        '''
        Move the signal along the axis by a certain distance
        '''
        self.data.iloc[:, 0] += distance

class ContinuousSignal1D(Signal1D):
    '''
    NumericSignal1D 是 Signal1D 的子类, 用于表示 axis 与 value 都是连续
    '''
    
    def __init__(
        self,
        data,
        name,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        update="to_data",
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
        axis_limit=None,
        axis_margin=(0, 0),
        value_limit=None,
        value_margin=(0, 0),
    ):
        super().__init__(
            data=data,
            name=name,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            update=update,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
        )
        if not axis_limit:
            axis_limit = (self.get_axis().min(), self.get_axis().max())
        if not value_limit:
            value_limit = (self.get_values().min(), self.get_values().max())
        # print(axis_margin)
        # print(value_margin)
        self.description_annotations = [
            ContDescAnno(
                self.get_axis_name(),
                self.get_axis_unit(),
                axis_limit,
                tick_number=10,
                margin=axis_margin,
            ),
            ContDescAnno(
                self.get_value_name(),
                self.get_value_unit(),
                value_limit,
                tick_number=10,
                margin=value_margin,
            ),
        ]

    def copy(self):
        copied_signal = type(self)(
            data=self.get_data().copy(),
            name=self.get_name(),
            axis_name=self.get_axis_name(),
            axis_unit=self.get_axis_unit(),
            value_name=self.get_value_name(),
            value_unit=self.get_value_unit(),
            update="to_data",
            detect_axis_name_and_unit=False,
            detect_value_name_and_unit=False,
            axis_limit=self.get_axis_limit(),
            value_limit=self.get_value_limit(),
        )

        return copied_signal
    
    def get_value_at_axis(self, axis):
        '''
        使用 1 次插值获取任意 axis 处的值
        '''
        return np.interp(axis, self.get_axis(), self.get_values())

    def get_value_limit(self):
        return self.get_limit(1)

    def set_value_limit(self, value_limit):
        self.set_limit(1, value_limit)
        
    def set_default_relative_value_limit(self):
        axis_range = self.get_axis_limit()
        values_in_axis_range = self.get_values_between(axis_range[0], axis_range[1])
        self.set_value_limit((values_in_axis_range.min(), values_in_axis_range.max()))
        
    def get_value_margin(self):
        return self.get_margin(1)
    
    def set_value_margin(self, value_margin):
        self.set_margin(1, value_margin)

    def __mul__(self, factor):
        """
        Multiply the values of the signal by the given factor.
        This method enables the * operator to be used for multiplication.
        """
        values = self.get_values()
        values *= factor

    def rescale_between(self, target_0, target_1, inplace=False):
        values = (self.get_values() - target_0) / (target_1 - target_0)
        if inplace:
            self.set_values(values)
        return values

    def get_peak_between(self, start, end) -> typing.Tuple[float, float]:
        """
        寻找当 axis 位于 start 与 end 之间时, value 的最大值以及对应的 axis, 返回一个 (axis, value) 的元组
        """
        peak_idx = self.get_values_between(start, end).idxmax()
        peak_axis = self.get_axis_between(start, end)[peak_idx]
        peak_value = self.get_values_between(start, end)[peak_idx]
        return (peak_axis, peak_value)

    def plot_at(self, ax: plt.Axes, **kwargs):
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
        kwargs.pop("text_shift", None)
        label = kwargs.pop("label", None)
        if not label:
            label = self.get_name()
        kwargs_for_annotate = kwargs.copy()
        kwargs_for_Line2D = kwargs.copy()
        
        kwargs_for_Line2D.pop("fontsize", None)
        kwargs_for_Line2D.pop("rotation", None)
        
        handle, = ax.plot(
            self.rescale_for_plots(
                self.get_axis(),
                self.get_axis_limit(),
                self.get_axis_margin(),
            ),
            self.rescale_for_plots(
                self.get_values(),
                self.get_value_limit(),
                self.get_value_margin(),
            ),
            label=label, **kwargs_for_Line2D
        )
        return handle

    def plot_peak_at(
        self, ax: plt.Axes, start, end, type="vline", text_shift=(0, 0), **kwargs
    ):
        """
        type can be "vline" or "annotation"
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
        axis_limit = self.get_axis_limit()
        value_limit = self.get_value_limit()
        rescaled_peak_axis = transform_utils.rescale_to_0_1(peak_axis, axis_limit[0], axis_limit[1])
        rescaled_peak_value = transform_utils.rescale_to_0_1(peak_value, value_limit[0], value_limit[1])
        if type == "vline":
            self._plot_peak_at_with_vline(
                ax,
                rescaled_peak_axis=rescaled_peak_axis,
                rescaled_peak_value=rescaled_peak_value,
                peak_axis=peak_axis,
                peak_value=peak_value,
                text_shift=text_shift,
                **kwargs
            )
        elif type == "annotation":
            self._plot_peak_at_with_annotation(
                ax,
                peak_axis=peak_axis,
                peak_value=peak_value,
                rescaled_peak_axis=rescaled_peak_axis,
                rescaled_peak_value=rescaled_peak_value,
                text_shift=text_shift,
                **kwargs
            )

    def _plot_peak_at_with_vline(
        self, ax: plt.Axes, rescaled_peak_axis, rescaled_peak_value, peak_axis, peak_value, text_shift, **kwargs
    ):
        linestyle = kwargs.pop("linestyle", "dashed")
        ax.vlines(rescaled_peak_axis, 0, rescaled_peak_value, linestyles=linestyle, **kwargs)
        ax.annotate(
            f"{peak_axis:.2f} {self.get_axis_unit()}",
            xy=(rescaled_peak_axis, rescaled_peak_value),
            xytext=(rescaled_peak_axis + text_shift[0], rescaled_peak_value + text_shift[1]),
            **kwargs,
        )

    def _plot_peak_at_with_annotation(
        self, ax: plt.Axes, rescaled_peak_axis, rescaled_peak_value, peak_axis, peak_value, text_shift, **kwargs
    ):
        ax.annotate(
            f"{peak_axis:.2f} {self.get_axis_unit()}",
            xy=(rescaled_peak_axis, rescaled_peak_value),
            xytext=(rescaled_peak_axis + text_shift[0], rescaled_peak_value + text_shift[1]),
            **kwargs,
        )
    
    def integrate_between(self, start, end):
        signal_data:pd.DataFrame = self.get_signal(signal)
        signal_data = signal_data[(signal_data.iloc[:, 0] >= start) & (signal_data.iloc[:, 0] <= end)]
        
        if not isinstance(baseline, Iterable):
            baseline = np.array([baseline for _ in range(len(signal_data))])
        else:
            if len(baseline) != len(signal_data):
                print("Baseline length should be equal to the length of signal data")
                return None
            else:
                baseline = np.array(baseline)
        
        signal_height = signal_data.iloc[:, 1] - baseline
        peak_area = np.trapz(signal_height, signal_data.iloc[:, 0])
        
        if ax:
            ax.vlines([start, end], 0, 1, colors=color, linestyles=linestyles, linewidths=linewidths, alpha=max(1, alpha*2))
            ax.fill_between(signal_data.iloc[:, 0],
                rescale_signal(baseline, self.y_limits[signal][0], self.y_limits[signal][1]),
                rescale_signal(signal_data.iloc[:, 1].copy(), self.y_limits[signal][0], self.y_limits[signal][1]),
                color=color, alpha=alpha
            )
        print("Peak area = {} {}·ml from {} ml to {} ml".format(peak_area, signal_data.columns[1], start, end))
        
        return peak_area

class DiscreteSignal1D(Signal1D):
    '''
    AnnotationSignal1D 是 Signal1D 的子类, 用于表示 axis 是数值, 但是 value 离散的信号
    '''
    @classmethod
    def from_csv(
        cls,
        path,
        name=None,
        detect_axis_name_and_unit=True,
        detect_value_name_and_unit=False,
        axis_name=None,
        axis_unit=None,
        value_name="Fraction",
        value_unit=None,
        **kwargs
    ):
        return super().from_csv(
            path,
            name,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            **kwargs)
    
    def __init__(
        self,
        data,
        name, 
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        update="to_data",
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
    ):
        super().__init__(
            data=data,
            name=name,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            update=update,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit
        )
        self.description_annotations = [
            ContDescAnno(
                self.get_axis_name(),
                self.get_axis_unit(),
                (self.get_axis().min(), self.get_axis().max()),
                10,
            ),
            DiscDescAnno(
                self.get_value_name(),
                self.get_value_unit()
            ),
        ]
        self.arrowprops = {}

    def plot_at(
        self, ax: plt.Axes, label=None, text_shift=(0, 0), mark_height=0.5, **kwargs
    ):
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
        if not label:
            label = self.get_name()
        
        kwargs_for_annotate = kwargs.copy()
        if not "rotation" in kwargs_for_annotate.keys():
            kwargs_for_annotate["rotation"] = 90 
        if not "fontsize" in kwargs_for_annotate.keys():
            kwargs_for_annotate["fontsize"] = 10
        
        kwargs_for_Line2D = kwargs.copy()
        kwargs_for_Line2D.pop("rotation", None)
        kwargs_for_Line2D.pop("fontsize", None)
        axis_to_plot = transform_utils.rescale_to_0_1(self.get_axis(), self.get_axis_limit()[0], self.get_axis_limit()[1])
        ax.vlines(axis_to_plot, 0, mark_height, linestyles="dashed", **kwargs_for_Line2D)
        for axis, value in zip(axis_to_plot, self.get_values()):
            ax.annotate(
                f"{value}",
                xy=(axis, mark_height),
                xytext=(axis + text_shift[0], 0.5 + text_shift[1]),
                ha="center",
                **kwargs_for_annotate,
            )

        handle = plt.Line2D(
            [0], [0], label=label, **kwargs_for_Line2D
        )  # Generate a virtual handle for legend
        return handle
    
    def set_arrowprops(self, arrowprops):
        '''
        Set the arrowprops for annotations. See ax.annotate for supported arrowprops
        '''
        self.arrowprops = arrowprops

class SegmentedSignal1D(Signal1D):
    
    def __init__(
        self,
        **kwargs
    ):
        '''
        A segmented signal 1d should look like\n
        Axis,Values,Segment,Others\n
        0.2,0.2,1,1\n
        0.22,0.3,1,2\n
        ...\n
        0.4,0.4,1,9\n
        0.38,0.38,2,10\n
        ...\n
        '''
        
        self.active_segments = "all"
    
    def get_segment_num(self):
        return set(self.data["Segment"])
    
    def get_active_segments(self):
        return self.active_segments
    
    def get_indices_for_segments(self, segments):
        if segments == "all":
            return self.data.index
        elif isinstance(segments, int):
            return self.data.index[self.data["Segment"] == segments]
        elif isinstance(segments, list):
            return self.data.index[self.data["Segment"].isin(segments)]
        else:
            raise ValueError(f"Segments should be 'all', int or list of int, but got {segments}, which is {type(segments)}")
    
    def get_indices_for_active_segments(self):
        return self.get_indices_for_segments(self.get_active_segments())
        
    def set_active_segments(self, segments):
        if isinstance(segments, str):
            if segments != "all":
                raise ValueError("segments should be 'all', int or list of int")
        elif isinstance(segments, int):
            if segments < 0 or segments > max(self.data["Segment"]):
                raise ValueError("segments should be positive int and less than the max segment number")
        elif isinstance(segments, list):
            if any([x < 0 or x > max(self.data["Segment"]) for x in segments]):
                raise ValueError("segments should be positive int and less than the max segment number")
        else:
            raise ValueError("segments should be 'all', int or list of int")
        self.active_segments = segments
        
    def get_data_for_segments(self, segments):
        # return self.get_data().loc[]
        if segments == "all":
            return self.data
        elif isinstance(segments, int):
            return self.data[self.data["Segment"] == segments]
        elif isinstance(segments, list):
            return self.data[self.data["Segment"].isin(segments)]
        else:
            raise ValueError(f"Segments should be 'all', int or list of int, but got {segments}, which is {type(segments)}")
    
    def get_data_for_active_segments(self):
        return self.get_data_for_segments(self.get_active_segments())
        
    def get_series_for_segements(self, segments, i):
        '''
        i = 0 for axis, i = 1 for values
        '''
        segement_data = self.get_data_for_segments(segments)
        return segement_data.iloc[:, i]
    
    def get_axis_for_segments(self, segments):
        return self.get_series_for_segements(segments, 0)
    
    def get_values_for_segments(self, segments):
        return self.get_series_for_segements(segments, 1)
    
    def get_series_for_active_segments(self, i):
        '''
        i = 0 for axis, i = 1 for values
        '''
        return self.get_series_for_segements(self.get_active_segments(), i)
    
    def get_axis_for_active_segments(self):
        return self.get_series_for_active_segments(0)
    
    def get_values_for_active_segments(self):
        return self.get_series_for_active_segments(1)
        
    def plot(self, ax: plt.Axes, color="C0"):
        
        def pplot(data):
            ax.plot(
                (data["Potential (V)"] - self.get_voltage_limit()[0]) / (self.get_voltage_limit()[1] - self.get_voltage_limit()[0]), 
                (data["Current (A)"] - self.get_current_limit()[0]) / (self.get_current_limit()[1] - self.get_current_limit()[0]), 
                label=self.get_name(), color=color
            )
        if self.active_segments == "all":
            pplot(self.data)
        elif isinstance(self.active_segments, int):
            segment_data = self.data[self.data["Segment"] == self.active_segments]
            pplot(segment_data)
        elif isinstance(self.active_segments, list):
            segment_data = self.data[self.data["Segment"].isin(self.active_segments)]
            pplot(segment_data)
        else:
            raise ValueError(f"Segments should be 'all', int or list of int, but got {self.active_segments}, which is {type(self.active_segments)}")

class SegmentedContinuousSignal1D(SegmentedSignal1D, ContinuousSignal1D):
    
    def __init__(
        self,
        data,
        name,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        update="to_data",
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
        axis_limit=None,
        axis_margin=(0.1, 0.1),
        axis_digits=11,
        value_limit=None,
        value_margin=(0.1, 0.1),
        value_digits=11,
    ):
        # print(axis_margin)
        # print(value_margin)
        ContinuousSignal1D.__init__(
            self,
            data=data,
            name=name,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            update=update,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
            axis_limit=axis_limit,
            axis_margin=axis_margin,
            value_limit=value_limit,
            value_margin=value_margin,
        )
        SegmentedSignal1D.__init__(self)
    
    def set_active_segments(self, segments, update_value_limit = True):
        res = super().set_active_segments(segments)
        if update_value_limit:
            self.set_default_relative_value_limit()
    
    def set_default_relative_value_limit(self):
        segment_indices = self.get_indices_for_active_segments()
        axis_indices = self.get_indices_for_current_axis_limit()
        indices = list(set(segment_indices) & set(axis_indices))
        indices.sort()
        my_data = self.get_data().loc[indices, :]
        value_limit = (my_data.iloc[:, 1].min(), my_data.iloc[:, 1].max())
        self.description_annotations[1].set_limit(value_limit)
    
    def set_default_limit_and_margin(self):
        # 分析 signal 的 axis 与 value 范围, 设定 axis 和 value 的默认范围
        
        segment_data = self.get_data_for_active_segments()
        
        axis_limit = (min(segment_data.iloc[:, 0]), max(segment_data.iloc[:, 0]))
        value_limit = (min(segment_data.iloc[:, 1]), max(segment_data.iloc[:, 1]))
        self.description_annotations[0].set_limit(axis_limit)
        self.description_annotations[1].set_limit(value_limit)
        
    def plot_at(self, ax: plt.Axes, **kwargs):
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
        kwargs.pop("text_shift", None)
        label = kwargs.pop("label", None)
        if not label:
            label = self.get_name()
            
        kwargs_for_annotate = kwargs.copy()
        kwargs_for_Line2D = kwargs.copy()
        
        kwargs_for_Line2D.pop("fontsize", None)
        kwargs_for_Line2D.pop("rotation", None)
        
        handle, = ax.plot(
            self.rescale_for_plots(
                self.get_axis_for_active_segments(),
                self.get_axis_limit(),
                self.get_axis_margin()
            ),
            self.rescale_for_plots(
                self.get_values_for_active_segments(),
                self.get_value_limit(),
                self.get_value_margin()
            ),
            label=label, **kwargs_for_Line2D
        )
        return handle

class FractionSignal(DiscreteSignal1D):
    @classmethod
    def from_csv(
        cls,
        path,
        name=None,
        detect_axis_name_and_unit=True,
        detect_value_name_and_unit=False,
        axis_name=None,
        axis_unit=None,
        value_name="Fraction",
        value_unit=None,
        **kwargs
    ):
        return super().from_csv(
            path=path,
            name=name,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            **kwargs
        ) 

    def __init__(
        self,
        data,
        name,
        axis_name=None,
        axis_unit=None,
        value_name="Fraction",
        value_unit=None,
        update="to_data",
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
    ):
        super().__init__(
            data=data,
            name=name,
            axis_name=axis_name,
            axis_unit=axis_unit,
            update=update,
            value_name=value_name,
            value_unit=value_unit,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
        )
        if not axis_unit in chromatographic_units_map_types.keys():
            raise ValueError(
                f"Expected axis_unit to be one of {chromatographic_units_map_types.keys()}, but got {axis_unit}"
            )

        if axis_name == "undefined":
            self.set_axis_name(chromatographic_units_map_types[axis_unit])
