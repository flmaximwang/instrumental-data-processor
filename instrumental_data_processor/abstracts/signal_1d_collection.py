from typing import Mapping
import numpy as np
from instrumental_data_processor.abstracts.signal_1d import Signal1D
from .signal_collection import SignalCollection
from .signal_1d import Signal1D, ContinuousSignal1D
import matplotlib.pyplot as plt

class Signal1DCollection(SignalCollection):
    
    display_modes = ["main_signal_axis", "all_axis", "separate", "denoted_axis"]
    
    def __init__(self, signals: list[Signal1D] = [], name="Default Signal1DCollection") -> None:
        for signal in signals:
            if not isinstance(signal, Signal1D):
                raise TypeError(f"Signal {signal} is not a Signal1D")
        self.signals: Mapping[str, Signal1D] = {}
        super().__init__(signals, name=name)
        self.set_default_parameters()
        
    def set_default_parameters(self) -> None:
        self.main_signal_name: str = list(self.signals.keys())[0]
        self.visible_signal_names: list[str] = list(self.signals.keys())
        self.display_mode: str = 0 # See display_modes for its meaning
        self.figsize = None
        main_signal = self.signals[self.main_signal_name]
        self.axis_limit = main_signal.get_axis_limit()
        self.axis_name = main_signal.get_axis_name()
        self.axis_unit = main_signal.get_axis_unit()
        self.axis_tick_number = 11
        
    def set_figsize(self, figsize):
        self.figsize = figsize
        
    def unset_figsize(self):
        self.figsize = None
        
    def subplots(self, nrow=1, ncol=1):
        if self.figsize:
            return plt.subplots(nrow, ncol, figsize=self.figsize)
        else:
            return plt.subplots(nrow, ncol)
        
    def get_signal(self, signal_name: str) -> Signal1D:
        return super().get_signal(signal_name)
    
    def __getitem__(self, signal_name: str):
        return self.get_signal(signal_name)

    def set_main_signal(self, main_signal_name, update_axis=True):
        self.main_signal_name = main_signal_name
        if update_axis:
            self.update_axis_name_and_unit_from_main_signal()
    
    def set_visible_signals(self, *signal_names):
        self.visible_signal_names = list(signal_names)
    
    def align_signal_axes(self, axis_limit) -> None:
        for signal in self.signals.keys():
            self.signals[signal].set_axis_limit(axis_limit)
        self.axis_limit = axis_limit
        
    def get_axis_name(self):
        return self.axis_name
    
    def set_axis_name(self, axis_name):
        self.axis_name = axis_name
        
    def get_axis_unit(self):
        return self.axis_unit
    
    def set_axis_unit(self, axis_unit):
        self.axis_unit = axis_unit
        
    def get_axis_label(self):
        if self.get_axis_unit() is None:
            return self.get_axis_name()
        else:
            return f"{self.get_axis_name()} ({self.get_axis_unit()})"
        
    def update_axis_name_and_unit_from_main_signal(self):
        main_signal = self[self.main_signal_name]
        self.set_axis_name(main_signal.get_axis_name())
        self.set_axis_unit(main_signal.get_axis_unit())
    
    def get_axis_limit(self):
        return self.axis_limit
    
    def set_axis_limit(self, axis_limit):
        self.axis_limit = axis_limit
        
    def set_axis_tick_number(self, axis_tick_number):
        self.axis_tick_number = axis_tick_number
        
    def get_axis_tick_number(self):
        return self.axis_tick_number
        
    def get_axis_ticks(self):
        return np.linspace(0, 1, self.get_axis_tick_number())
    
    def get_axis_ticklabels(self):
        lower_limit, upper_limit = self.get_axis_limit()
        tick_number = self.get_axis_tick_number()
        return np.round(np.linspace(lower_limit, upper_limit, tick_number), 1)
        
    def set_display_mode(self, display_mode):
        self.display_mode = display_mode
        
    def plot_with_main_signal_value_label(self, **kwargs):
        
        ax: plt.Axes
        fig, ax = self.subplots(1, 1)
        axes = [ax]
        handles = []
        for i, signal_name in enumerate(self.visible_signal_names):
            signal = self.signals[signal_name]
            handles.append(signal.plot_at(ax, color=f"C{i}", **kwargs))
        main_signal = self[self.main_signal_name]
        xticks, xticklabels, yticks, yticklabels = self.get_axis_ticks(), self.get_axis_ticklabels(), main_signal.get_value_ticks(), main_signal.get_value_tick_labels()
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(self.get_axis_label())
        ax.set_ylabel(main_signal.get_value_label())
        ax.legend(handles = handles)
        ax.set_title(self.get_name())
        fig.tight_layout()
        return fig, axes
    
    def plot_with_all_value_labels(self, axis_shift, **kwargs):
        ax: plt.Axes
        fig, ax = self.subplots(1, 1)
        twins: list[plt.Axes] = []
        handles: list[plt.Line2D] = []
        counter = 0
        for i, signal_name in enumerate(self.visible_signal_names):
            signal = self.signals[signal_name]
            tick_weight = dict(size=4, width=1.5)
            print(signal_name)
            if i > 0:
                print(1)
                if isinstance(self.signals[signal_name], ContinuousSignal1D):
                    twins.append(ax.twinx())
                    twins[-1].spines.right.set_position(("axes", 1 + axis_shift * counter))
                    counter += 1
                    ax_to_plot = twins[-1]
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                    ax_to_plot.set_xlim([0, 1])
                    ax_to_plot.set_ylim([0, 1])
                    
                    ax_to_plot.set_ylabel(self.signals[signal_name].get_value_label())
                    ax_to_plot.yaxis.label.set_color(handle.get_color())
                    
                    ax_to_plot.tick_params(axis='y', colors=handle.get_color(), **tick_weight)
                    yticks, yticklabels = signal.get_value_ticks(), signal.get_value_tick_labels()
                    ax_to_plot.set_yticks(yticks)
                    ax_to_plot.set_yticklabels(yticklabels)
                else:
                    ax_to_plot = ax
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
            else:
                if isinstance(self.signals[signal_name], ContinuousSignal1D):
                    ax_to_plot = ax
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                    ax_to_plot.set_xlim(self.get_axis_limit())
                    ax_to_plot.set_ylim([0, 1])
                    ax_to_plot.set_ylabel(self.signals[signal_name].get_value_label())
                    ax_to_plot.yaxis.label.set_color(handle.get_color())
                    
                    ax_to_plot.tick_params(axis='y', colors=handle.get_color(), **tick_weight)
                    yticks, yticklabels = self.signals[signal_name].get_value_ticks(), self.signals[signal_name].get_value_tick_labels()
                    ax_to_plot.set_yticks(yticks)
                    ax_to_plot.set_yticklabels(yticklabels)
                else:
                    raise TypeError("The first signal should be a ContinuousSignal1D")
            handles.append(handle)    
        ax.set_xlabel(self.get_axis_label())
        xticks, xticklabels = self.get_axis_ticks(), self.get_axis_ticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(handles=handles)
        ax.set_title(self.get_name())
        fig.tight_layout()
        # 避免右侧额外的坐标轴跑到画布以外
        fig.subplots_adjust(right=1 - axis_shift * counter)
        return fig, [ax] + twins
    
    def plot_separately(self) -> tuple[plt.Figure, list[plt.Axes]]:
        return plt.subplots(1, 1)
    
    def plot_with_denoted_axis(self) -> tuple[plt.Figure, list[plt.Axes]]:
        return plt.subplots(1, 1)
        
    def plot(self, **kwargs) -> None:
        axes: list[plt.Axes]
        if self.display_mode == 0:
            fig, axes = self.plot_with_main_signal_value_label(**kwargs)
        elif self.display_mode == 1:
            axis_shift = kwargs.pop("axis_shift", 0.2)
            fig, axes = self.plot_with_all_value_labels(axis_shift=axis_shift, **kwargs)
        elif self.display_mode == 2:
            fig, axes = self.plot_separately()
        elif self.display_mode == 3:
            fig, axes = self.plot_with_denoted_axis()
        else:
            raise Exception("Unknown display mode")

        return fig, axes