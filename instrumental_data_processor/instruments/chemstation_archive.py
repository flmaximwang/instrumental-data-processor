'''
Author: Fanlin Maxim Wang
Date: 2023-12-31
Description: This module provides some useful functions for chromatography data processing.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import os
from collections import OrderedDict
from typing import Mapping, Iterable

def rescale_signal(signal_intensity: float | np.ndarray, target_min: float, target_max: float):
    """
    线性缩放 signal_intensity 到 target_min 和 target_max 之间
    
    - signal_intensity: 信号强度, 可以是一个 float, 也可以是一个 numpy.ndarray
    - target_min: 被缩放到 0
    - target_max: 被缩放到 1
    """

    result = (signal_intensity - target_min) / (target_max - target_min)
    return result

class Signal:
    '''
    通用的一维信号类，信号会对某一尺度进行标注，例如时间、体积、波长等
    '''
    progress_units_map_types = {
        "ml": "Volume",
        "min": "Time",
    }
    
    def __init__(self, data: str | pd.DataFrame, name="", encoding="utf-8", sep=",", progress_unit = "ml", silent = False):
        """
        Construct a signal from a file or a pandas.DataFrame.
        - data: a file path or a pandas.DataFrame
        - name: the name of the signal
        """
        if isinstance(data, str):
            if not name:
                self.name = ".".join(os.path.basename(data).split(".")[:-1])
            self.data = data = pd.read_csv(data, encoding=encoding, sep=sep)
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) != 2:
                raise ValueError(
                    "Signal data should have 2 columns, but got {} columns".format(
                        len(data.columns)
                    )
                )
            self.name = name
            self.data = data
            
        self.progress_unit = progress_unit
        self.progress_type = Signal.progress_units_map_types[progress_unit]
        if not silent:
            self.check_progress_type()
        self.xlim = (0, 0)
        self.set_default_xlim()
        self.x_tick_number = 11
    
    def get_progress_unit(self):
        return self.progress_unit
    
    def get_progress_type(self):
        if not self.progress_type == NumericSignal.progress_units_map_types[self.progress_unit]:
            raise ValueError("Progress type should be in {}".format(list(NumericSignal.progress_units_map_types.keys())))
        return self.progress_type
    
    def set_progress_unit(self, progress_unit):
        if not progress_unit in NumericSignal.progress_units_map_types.keys():
            raise ValueError("Progress type should be in {}".format(list(NumericSignal.progress_types())))
        self.progress_type = (progress_unit, NumericSignal.progress_units_map_types[progress_unit])
        
    def check_progress_type(self):
        print("Annotated progress type and units from data: {}".format(self.get_progresses().name))
        print("Current progress type: {}".format(self.get_progress_type()))
        print("Current progress unit: {}".format(self.get_progress_unit()))

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_progresses(self):
        return self.get_data().iloc[:, 0]
    
    def get_annotations(self):
        return self.get_data().iloc[:, 1]

    def set_default_xlim(self):
        self.xlim = (min(self.data.iloc[:, 0]), max(self.data.iloc[:, 0]))
        
    def set_xlabel(self, new_xlabel=None):
        old_ylabel = self.get_data().columns[0]
        if not new_xlabel:
            new_xlabel = self.progress_type + " (" + self.progress_unit + ")"
        self.get_data().rename(columns={old_ylabel: new_xlabel}, inplace=True)
        
    def set_ylabel(self, new_ylabel):
        old_ylabel = self.get_data().columns[1]
        self.get_data().rename(columns={old_ylabel: new_ylabel}, inplace=True)

    def set_xlim(self, *xlim):
        self.xlim = xlim

    def get_xlim(self):
        return self.xlim

    def set_x_tick_number(self, x_tick_number):
        self.x_tick_number = x_tick_number

    def get_x_ticks(self):
        return np.linspace(self.xlim[0], self.xlim[1], self.x_tick_number), np.round(
            np.linspace(self.xlim[0], self.xlim[1], self.x_tick_number), 1
        )

    def get_data_between_progress(self, start, end):
        return self.get_data()[
            (self.get_progresses() >= start) & (self.get_progresses() <= end)
        ]

    def get_progresses_between_progress(self, start, end):
        return self.get_progresses()[
            (self.get_progresses() >= start) & (self.get_progresses() <= end)
        ]
        
    def plot_at(
        self,
        ax: plt.Axes,
        color="black",
        alpha=1,
        linewidth=1,
        label=None,
        linestyle="dashed",
        fontsize=5,
        rotation=90
    ):
        label = self.get_name() if not label else label
        ax.vlines(self.get_progresses(), 0, 0.5, color=color, alpha=alpha, linewidth=linewidth, label=label, linestyle=linestyle)
        for progress, annotation in zip(self.get_progresses(), self.get_annotations()):
            ax.annotate(annotation, (progress, 0.5), rotation=rotation, fontsize=fontsize)
        handle = Line2D([0], [0], color=color, alpha=alpha, linewidth=linewidth, label=label, linestyle=linestyle)
        return [handle]
    
    def preview(self, rotation = 90, fontsize = 5):
        fig, ax = plt.subplots(1, 1)
        ax.vlines(self.get_progresses(), [0 for _ in range(len(self.get_progresses()))], [0.5 for _ in range(len(self.get_progresses()))])
        for progress, annotation in zip(self.get_progresses(), self.get_annotations()):
            ax.annotate(annotation, (progress, 0.5), rotation=rotation, fontsize=fontsize)
        ax.set_xlim(self.xlim)
        ax.set_ylim([0, 1])
        ax.set_title(self.get_name())
        ax.set_xlabel(self.get_progresses().name)
        ax.set_ylabel(self.get_annotations().name)
        xticks, xticklabels = self.get_x_ticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        return fig, ax

class NumericSignal(Signal):
    
    def __init__(self, data: str | pd.DataFrame, name="", encoding="utf-8", sep=",", progress_unit = "ml", silent = False):
        """
        Construct a signal from a file or a pandas.DataFrame.
        - data: a file path or a pandas.DataFrame
        - name: the name of the signal
        """
        if isinstance(data, str):
            if not name:
                self.name = ".".join(os.path.basename(data).split(".")[:-1])
            self.data = data = pd.read_csv(data, encoding=encoding, sep=sep)
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) != 2:
                raise ValueError(
                    "Signal data should have 2 columns, but got {} columns".format(
                        len(data.columns)
                    )
                )
            self.name = name
            self.data = data
            
        self.progress_unit = progress_unit
        self.progress_type = NumericSignal.progress_units_map_types[progress_unit]
        if not silent:
            self.check_progress_type()
        self.xlim = (0, 0)
        self.ylim = (0, 0)
        self.set_default_xlim()
        self.set_default_ylim()
        self.x_tick_number = 11
        self.y_tick_number = 11
    
    def get_progress_unit(self):
        return self.progress_unit
    
    def get_progress_type(self):
        if not self.progress_type == NumericSignal.progress_units_map_types[self.progress_unit]:
            raise ValueError("Progress type should be in {}".format(list(NumericSignal.progress_units_map_types.keys())))
        return self.progress_type
    
    def set_progress_unit(self, progress_unit):
        if not progress_unit in NumericSignal.progress_units_map_types.keys():
            raise ValueError("Progress type should be in {}".format(list(NumericSignal.progress_types())))
        self.progress_type = (progress_unit, NumericSignal.progress_units_map_types[progress_unit])
        
    def check_progress_type(self):
        print("Annotated progress type and units from data: {}".format(self.get_progress().name))
        print("Current progress type: {}".format(self.get_progress_type()))
        print("Current progress unit: {}".format(self.get_progress_unit()))

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_progress(self):
        return self.get_data().iloc[:, 0]

    def get_intensity(self):
        return self.get_data().iloc[:, 1]

    def set_default_xlim(self):
        self.xlim = (min(self.data.iloc[:, 0]), max(self.data.iloc[:, 0]))
        
    def set_xlabel(self, new_xlabel=None):
        old_ylabel = self.get_data().columns[0]
        if not new_xlabel:
            new_xlabel = self.progress_type + " (" + self.progress_unit + ")"
        self.get_data().rename(columns={old_ylabel: new_xlabel}, inplace=True)
        
    def set_ylabel(self, new_ylabel):
        old_ylabel = self.get_data().columns[1]
        self.get_data().rename(columns={old_ylabel: new_ylabel}, inplace=True)

    def set_xlim(self, *xlim):
        self.xlim = xlim

    def get_xlim(self):
        return self.xlim

    def set_default_ylim(self):
        self.ylim = (min(self.data.iloc[:, 1]), max(self.data.iloc[:, 1]))

    def set_ylim(self, *ylim):
        self.ylim = ylim

    def get_ylim(self):
        return self.ylim

    def set_x_tick_number(self, x_tick_number):
        self.x_tick_number = x_tick_number

    def get_x_ticks(self):
        return np.linspace(self.xlim[0], self.xlim[1], self.x_tick_number), np.round(
            np.linspace(self.xlim[0], self.xlim[1], self.x_tick_number), 1
        )

    def set_y_tick_number(self, y_tick_number):
        self.y_tick_number = y_tick_number

    def get_y_ticks(self):
        return np.linspace(0, 1, self.y_tick_number), np.round(
            np.linspace(self.ylim[0], self.ylim[1], self.y_tick_number), 1
        )

    def rescale_signal(self, target_min, target_max, inplace=False):
        """
        target_min 会被缩放到 0, target_max 会被缩放到 1
        signal_data 会根据以上的规则线性缩放
        """

        result = (self.get_intensity() - target_min) / (target_max - target_min)
        if inplace:
            self.get_data().iloc[:, 1] = result
        return result

    def get_data_between_progress(self, start, end):
        return self.get_data()[
            (self.get_progress() >= start) & (self.get_progress() <= end)
        ]

    def get_progress_between_progress(self, start, end):
        return self.get_progress()[
            (self.get_progress() >= start) & (self.get_progress() <= end)
        ]

    def get_intensity_between_progress(self, start, end):
        return self.get_intensity()[
            (self.get_progress() >= start) & (self.get_progress() <= end)
        ]

    def get_peak_between(
        self,
        start,
        end,
        ax: plt.Axes = None,
        color="black",
        alpha=0.1,
        linestyles="dashed",
        linewidth=0.5,
        fontsize=5,
        rotation=90,
        text_shift=(0, 0.05),
    ):
        peak_progress_idx = self.get_intensity_between_progress(start, end).idxmax()
        peak_intensity = self.get_intensity_between_progress(start, end).max()
        peak_progress = self.get_progress_between_progress(start, end)[
            peak_progress_idx
        ]
        if ax:
            ax.annotate(
                f"{peak_progress:.2f}",
                (
                    peak_progress,
                    rescale_signal(
                        peak_intensity, self.get_ylim()[0], self.get_ylim()[1]
                    ),
                ),
                (
                    peak_progress + text_shift[0],
                    rescale_signal(
                        peak_intensity, self.get_ylim()[0], self.get_ylim()[1]
                    ) + text_shift[1],
                ),
                color=color,
                fontsize=fontsize,
                rotation=rotation,
                arrowprops=dict(arrowstyle="-", color=color, linewidth=linewidth),
            )
        return peak_progress

    def integrate_signal_between(
        self,
        start,
        end,
        ax: plt.Axes = None,
        color="red",
        alpha=0.1,
        linestyles="dashed",
        linewidths=0.5,
        baseline=0.0,
    ):
        """
        start 和 end 为进度值，baseline 为基线值
        """

        signal_data: pd.DataFrame = self.get_data()[
            (self.get_progress() >= start) & (self.get_progress() <= end)
        ]
        if not isinstance(baseline, Iterable):
            baseline = np.array([baseline for _ in range(len(signal_data))])
        else:
            if len(baseline) != len(signal_data):
                raise ValueError(
                    "Baseline length should be equal to the length of signal data"
                )
            else:
                baseline = np.array(baseline)

        signal_height = signal_data.iloc[:, 1] - baseline
        peak_area = np.trapz(signal_height, signal_data.iloc[:, 0])

        if ax:
            ax.vlines(
                [start, end],
                0,
                1,
                colors=color,
                linestyles=linestyles,
                linewidths=linewidths,
                alpha=max(1, alpha * 2),
            )
            ax.fill_between(
                signal_data.iloc[:, 0],
                rescale_signal(baseline, self.ylim[0], self.ylim[1]),
                rescale_signal(
                    signal_data.iloc[:, 1], self.get_ylim()[0], self.get_ylim()[1]
                ),
                color=color,
                alpha=alpha,
            )
        print(
            "Peak area = {0} {1}·{2} from {3} {2} to {4} {2}".format(
                peak_area, signal_data.columns[1], self.get_progress_unit(), start, end
            )
        )

        return peak_area

    def plot_at(
        self,
        ax: plt.Axes,
        color="black",
        alpha=1,
        linewidth=1,
        label=None,
        linestyle="solid",
    ):
        rescaled_data = self.rescale_signal(self.ylim[0], self.ylim[1], inplace=False)
        label = self.get_name() if not label else label
        catch = ax.plot(
            self.get_data().iloc[:, 0],
            rescaled_data,
            label=label,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        return catch

    def preview(self):
        rescaled_data = self.rescale_signal(self.ylim[0], self.ylim[1], inplace=False)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.get_data().iloc[:, 0], rescaled_data)
        ax.set_xlim(self.xlim)
        ax.set_ylim([0, 1])
        ax.set_title(self.get_name())
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        xticks, xticklabels = self.get_x_ticks()
        yticks, yticklabels = self.get_y_ticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        return fig, ax
    
    def export(self, directory, encoding="utf-8", sep=","):
        self.get_data().to_csv(os.path.join(directory, self.get_name() + ".csv"), encoding=encoding, index=False, sep=sep)

    def copy(self):
        return NumericSignal(self.data.copy(), self.name, progress_unit=self.progress_unit, silent=True)


class Chromatogram:
    def __init__(self, data: str | list[NumericSignal], name=None, encoding="utf-8", sep=",", progress_unit="ml", silent=False):
        """
        Construct a chromatogram from a directory or a dict of signals.
        - data: a directory or a list of signals
        - name: the name of the chromatogram
        """

        self.signals = Mapping[str, NumericSignal]
        if isinstance(data, str):
            if not os.path.isdir(data):
                raise ValueError(
                    "Provided data should be a directory or a dict of signals like {'signal_name': signal_data}, but got {}".format(
                        type(data)
                    )
                )
            if data.endswith("/"):
                data = data[:-1]
            signals = {}
            for file in os.listdir(data):
                if file.endswith(".csv") or file.endswith(".CSV"):
                    file_path = os.path.join(data, file)
                    signal = NumericSignal(file_path, encoding=encoding, sep=sep, progress_unit=progress_unit, silent=silent)
                    signals[signal.get_name()] = signal
            self.signals = signals
            if not name:
                self.name = os.path.basename(data)
            else:
                self.name = name
        elif isinstance(data, list):
            if all(isinstance(signal, Signal) for signal in data):
                self.signals = dict(
                    (signal.get_name(), signal) for signal in data
                )
                self.name = name
            else:
                raise TypeError("Provided data should be a list of {} objects, but got {}".format(Signal, type(data)))
        else:
            raise TypeError(
                "Provided data should be a directory or a dict of signals like {'signal_name': signal_data}, but got {}".format(
                    type(data)
                )
            )

        self.visible_signals = list(self.signals.keys())

        self.xlim = (0, 0)
        self.set_default_xlim()
        self.progress_unit = progress_unit
        self.progress_type = NumericSignal.progress_units_map_types[progress_unit]

        self.main_signal = list(self.signals.keys())[0]
        self.xlabel = f"{self.progress_type} ({self.progress_unit})"
        self.x_tick_number = 11
        self.plot_fraction_flag = False
        self.figsize = None

    def get_available_signals(self):
        return list(self.signals.keys())

    def get_signal(self, signal: str = ""):
        if signal == "":
            print("Available signals: {}".format(self.get_available_signals()))
            return None
        if not signal in self.signals:
            print("No such signal: {}".format(signal))
            print("Available signals: {}".format(self.get_available_signals()))
            return None
        return self.signals[signal]

    def __getitem__(self, signal_name):
        return self.get_signal(signal_name)

    def set_signal(self, signal_name, signal_data):
        if not isinstance(signal_data, Signal):
            raise Exception("signal_data should be a Signal object")
        if signal_name in self.signals:
            raise AttributeError("Signal {} already exists. Please provide signal names other than: {}".format(signal_name, self.signals))
        self.signals[signal_name] = signal_data

    def __setitem__(self, signal_name, signal_data):
        self.set_signal(signal_name, signal_data)

    def delete_signal(self, signal_name):
        if not signal_name in self.signals:
            print("No such signal: {}".format(signal_name))
            print("Available signals: {}".format(list(self.signals.keys())))
            return None
        del self.signals[signal_name]

    def __delitem__(self, signal_name):
        self.delete_signal(signal_name)

    def rename_signal(self, old_signal_name, new_signal_name):
        if not old_signal_name in self.signals:
            print("No such signal: {}".format(old_signal_name))
            print("Available signals: {}".format(list(self.signals.keys())))
            return None
        self.get_signal(old_signal_name).set_name(new_signal_name)
        self.set_signal(new_signal_name, self.get_signal(old_signal_name))
        self.delete_signal(old_signal_name)
        if old_signal_name == self.main_signal:
            self.set_main_signal(new_signal_name)
        if old_signal_name in self.visible_signals:
            self.visible_signals.remove(old_signal_name)
            self.visible_signals.append(new_signal_name)

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_default_xlim(self):
        """
        选择 signal 中 xlim[0] 最小的作为 x_lim[0], xlim[1] 最大的作为 x_lim[1]
        """
        my_min = 0
        my_max = 0
        for signal_name in self.signals:
            my_min = min(self.signals[signal_name].get_xlim()[0], my_min)
            my_max = max(self.signals[signal_name].get_xlim()[1], my_max)
        self.xlim = [my_min, my_max]

    def set_xlim(self, my_min, my_max):
        self.xlim = [my_min, my_max]

    def get_xlim(self):
        return self.xlim

    def set_figsize(self, figsize_width, figsize_height):
        self.figsize = (figsize_width, figsize_height)

    def unset_figsize(self):
        self.figsize = None

    def set_visible_signals(self, *signals):
        self.visible_signals = signals

    def set_main_signal(self, signal):
        self.main_signal = signal

    def set_x_label(self, label):
        self.xlabel = label

    def get_x_ticks(self):
        xticks = np.linspace(self.xlim[0], self.xlim[1], self.x_tick_number)
        xticklabels = np.round(xticks, 1)
        return xticks, xticklabels

    def plot_signals(
        self,
        title=None,
        **kwargs,
    ):
        if self.figsize:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig, ax = plt.subplots(1, 1)
        for i, signal in enumerate(self.visible_signals):
            self.signals[signal].plot_at(ax, color=f"C{i}", **kwargs)
            
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.signals[self.main_signal].get_intensity().name)
        ax.set_xlim(self.xlim)
        ax.set_ylim([0, 1])
        yticks, yticklabels = self.signals[self.main_signal].get_y_ticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        xticks, xticklabels = self.get_x_ticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(self.visible_signals)
        ax.set_title(title)
        return fig, ax
    
    def plot_signals_with_all_yaxis(
        self,
        title=None,
        axis_shift = 0.2,
        **kwargs,
    ):
        if self.figsize:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig, ax = plt.subplots(1, 1)
        twins: list[plt.Axes] = []
        handles = []
        counter = 0
        for i, signal in enumerate(self.visible_signals):
            tkw = dict(size=4, width=1.5)
            if i > 0:
                if isinstance(self.signals[signal], NumericSignal):
                    twins.append(ax.twinx())
                    twins[-1].spines.right.set_position(("axes", 1 + axis_shift * counter))
                    counter += 1
                    ax_to_plot = twins[-1]
                    handle, = self.signals[signal].plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                    ax_to_plot.set_xlim(self.get_xlim())
                    ax_to_plot.set_ylim([0, 1])
                    ax_to_plot.set_ylabel(self.signals[signal].get_annotations().name)
                    ax_to_plot.yaxis.label.set_color(handle.get_color())
                    ax_to_plot.tick_params(axis='y', colors=handle.get_color(), **tkw)
                    yticks, yticklabels = self.signals[signal].get_y_ticks()
                    ax_to_plot.set_yticks(yticks)
                    ax_to_plot.set_yticklabels(yticklabels)
                else:
                    ax_to_plot = ax
                    handle, = self.signals[signal].plot_at(ax_to_plot, color=f"C{i}", **kwargs)
            else:
                if isinstance(self.signals[signal], NumericSignal):
                    ax_to_plot = ax
                    handle, = self.signals[signal].plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                    ax_to_plot.set_xlim(self.get_xlim())
                    ax_to_plot.set_ylim([0, 1])
                    ax_to_plot.set_ylabel(self.signals[signal].get_annotations().name)
                    ax_to_plot.yaxis.label.set_color(handle.get_color())
                    ax_to_plot.tick_params(axis='y', colors=handle.get_color(), **tkw)
                    yticks, yticklabels = self.signals[signal].get_y_ticks()
                    ax_to_plot.set_yticks(yticks)
                    ax_to_plot.set_yticklabels(yticklabels)
                else:
                    raise TypeError("The first signal should be a NumericSignal")
            handles.append(handle)

        ax.set_xlabel(self.xlabel)
        xticks, xticklabels = self.get_x_ticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(handles=handles)
        ax.set_title(title)
        return fig, ax

    def read_signal_from_file(
        self, file_path, signal_name="", encoding="utf-8", sep=","
    ):
        new_signal = NumericSignal(file_path, encoding=encoding, sep=sep)
        if not signal_name:
            signal_name = new_signal.get_name()
        if signal_name in self.signals:
            print("Signal {} already exists".format(signal_name))
            print("Please provide signal names other than: {}".format(self.signals))
            return None
        self.signals[signal_name] = new_signal

    def export(self, directory="."):
        directory = os.path.join(directory, self.get_name())
        if not os.path.exists(directory):
            os.makedirs(directory)
        for signal in self.signals:
            signal_name = self.signals[signal].get_name()
            self.signals[signal].export(directory)

import chromatography_utils
import pandas as pd
import os

class ChemStationChromatogram(chromatography_utils.Chromatogram):
    
    def __init__(self, directory):
        signals = []
        for file in os.listdir(directory):
            if not (file.endswith(".csv") or file.endswith(".CSV")):
                continue
            if file in ["Fraction.csv", "fraction.csv", "Fraction.CSV", "fraction.CSV"]:
                fraction_data = pd.read_csv(os.path.join(directory, file), encoding="utf-8", sep="\t")
                signal_data = pd.DataFrame({
                    "Time (ml)": [],
                    "Fraction": []
                })
                for index, fraction in fraction_data.iterrows():
                    row1 = pd.DataFrame({
                        "Time (ml)": [fraction["Start"]],
                        "Fraction": [fraction["AFC Loc"]]
                    })
                    row2 = pd.DataFrame({
                        "Time (ml)": [fraction["End"]],
                        "Fraction": "waste"
                    })
                    signal_data = pd.concat([signal_data, row1, row2], ignore_index=True)
                signal = chromatography_utils.Signal(signal_data, "Fraction", progress_unit = "min", silent=True)
                signal.set_name("Fraction")
            else:
                signal_data = pd.read_csv(os.path.join(directory, file), header=None, encoding="utf-16 LE", sep="\t")
                signal = chromatography_utils.NumericSignal(signal_data, progress_unit = "min", silent=True)
                signal.set_name(".".join(file.split(".")[:-1]))
                signal.set_ylabel(".".join(file.split(".")[:-1]))
            signal.set_xlabel()
            signals.append(signal)

        super().__init__(signals, name = os.path.basename(directory), silent=True, progress_unit="min")
