from typing import Mapping
import numpy as np
import pandas as pd
from .signal_collection import SignalCollection
from .signal_1d import Signal1D, ContinuousSignal1D
from .signal import ContDescAnno, DiscDescAnno
import matplotlib.pyplot as plt

class Signal1DCollection(SignalCollection):
    
    display_modes = ["main_signal_axis", "all_axis", "separate", "denoted_axis"]
    
    @staticmethod
    def merge(signal_collections: list['Signal1DCollection'], name="Merged Signal1DCollection", signal_renaming = True) -> 'Signal1DCollection':
        signals = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals.values():
                if signal_renaming:
                    signals.append(signal)
                    signal.set_name(f"{signal_collection.get_name()}_{signal.get_name()}")
        return Signal1DCollection(signals, name=name)
    
    def __init__(
            self,
            signals: list[Signal1D] = [],
            name="Default Signal1DCollection",
            main_signal_name: str = None,
            visible_signal_names: list[str] = None,
            display_mode: str = None,
            figsize=None,
            axis_description: ContDescAnno = None,
            value_description: ContDescAnno or DiscDescAnno = None,
        ) -> None:
        for signal in signals:
            if not isinstance(signal, Signal1D):
                raise TypeError(f"Signal {signal} is not a Signal1D")
        self.signals: Mapping[str, Signal1D] = {}
        super().__init__(signals, name=name)
        
        self.main_signal_name: str = list(self.signals.keys())[0] if not main_signal_name else main_signal_name
        self.visible_signal_names: list[str] = list(self.signals.keys()) if not visible_signal_names else visible_signal_names
        self.display_mode: str = 0 if not display_mode else display_mode # See display_modes for its meaning
        self.figsize = figsize
        
        if not axis_description:
            main_signal = self.signals[self.main_signal_name]
            axis_limit = main_signal.get_axis_limit()
            axis_name = main_signal.get_axis_name()
            axis_unit = main_signal.get_axis_unit()
            axis_tick_number = 11
            self.axis_description = ContDescAnno(axis_name, axis_unit, axis_limit, axis_tick_number)
        else:
            self.axis_description = axis_description.copy()
            
        if not value_description:
            main_signal = self.signals[self.main_signal_name]
            value_name = main_signal.get_value_name()
            value_unit = main_signal.get_value_unit()
            if isinstance(main_signal, ContinuousSignal1D):
                value_limit = main_signal.get_value_limit()
                value_tick_number = 11
                self.value_description = ContDescAnno(value_name, value_unit, value_limit, value_tick_number)
            else:
                self.value_description = DiscDescAnno(value_name, value_unit)
        else:
            self.value_description = value_description
        
        self.set_default_real_axis_limit()
        self.align_signal_axes(self.get_axis_limit())
    
    def __delitem__(self, sig_name):
        super().__delitem__(sig_name)
        if sig_name in self.visible_signal_names:
            self.visible_signal_names.remove(sig_name)
        if self.main_signal_name == sig_name:
            self.set_main_signal(self.visible_signal_names[0])
    
    def remove_signal(self, signal_name):
        self.__delitem__(signal_name)
        
    def get_main_signal(self):
        return self[self.main_signal_name]
    
    def get_axis_label(self):
        return self.axis_description.get_label()
    
    def get_axis_name(self):
        return self.axis_description.get_name()
    
    def set_axis_name(self, axis_name):
        return self.axis_description.set_name(axis_name)
    
    def get_axis_unit(self):
        return self.axis_description.get_unit()
    
    def set_axis_unit(self, axis_unit):
        return self.axis_description.set_unit(axis_unit)
    
    def get_axis_limit(self):
        return self.axis_description.get_limit()
    
    def set_axis_limit(self, axis_limit):
        return self.axis_description.set_limit(axis_limit)
    
    def get_axis_margin(self):
        return self.axis_description.get_margin()
    
    def set_axis_margin(self, margin):
        return self.axis_description.set_margin(margin)
    
    def get_axis_ticks(self):
        return self.axis_description.get_ticks()
    
    def get_axis_ticklabels(self, digits=1):
        return self.axis_description.get_ticklabels()
    
    def get_axis_tick_number(self):
        return self.axis_description.get_tick_number()
    
    def set_axis_tick_number(self, tick_number):
        return self.axis_description.set_tick_number(tick_number)

    def get_value_limit(self):
        return self.value_description.get_limit()

    def get_value_label(self):
        return self.value_description.get_label()
    
    def get_value_name(self):
        return self.value_description.get_name()
    
    def set_value_name(self, value_name):
        return self.value_description.set_name(value_name)
    
    def get_value_unit(self):
        return self.value_description.get_unit()
    
    def set_value_unit(self, value_unit):
        return self.value_description.set_unit(value_unit)
    
    def set_real_value_limit(self, value_limit):
        '''
        为 collection 中的每 1 个连续信号设置相同的 value_limit
        '''
        
        for sig_name in self:
            sig_if = self[sig_name]
            if isinstance(sig_if, ContinuousSignal1D):
                self[sig_name].set_value_limit(value_limit)
            else:
                pass
    
    def set_default_real_value_limit(self):
        
        my_limit = None
        for sig_name in self:
            sig_if = self[sig_name]
            if isinstance(sig_if, ContinuousSignal1D):
                if my_limit is None:
                    my_limit = sig_if.get_value_limit()
                else:
                    my_limit = (min(my_limit[0], sig_if.get_value_limit()[0]), max(my_limit[1], sig_if.get_value_limit()[1]))
            else:
                pass
        self.set_real_value_limit(my_limit)
    
    def set_default_relative_value_limit(self):
        
        for sig_name in self:
            sig_if = self[sig_name]
            if isinstance(sig_if, ContinuousSignal1D):
                sig_if.set_default_relative_value_limit()
            else:
                pass
    
    def set_formal_value_limit(self, value_limit):
        '''
        设置整个 collection 的表观 value limit,
        常用为每个信号设置好各自的 value limit, 然后进行对比, 并对整体标注 "相对值“
        '''
        return self.value_description.set_limit(value_limit)
    
    def get_value_ticks(self):
        return self.value_description.get_ticks()
    
    def get_value_ticklabels(self, digits=1):
        return self.value_description.get_ticklabels()
    
    def get_value_tick_number(self):
        return self.value_description.get_tick_number()
    
    def set_value_tick_number(self, tick_number):
        return self.value_description.set_tick_number(tick_number)
    
    def update_axis_name_and_unit_from_main_signal(self):
        main_signal = self[self.main_signal_name]
        self.set_axis_name(main_signal.get_axis_name())
        self.set_axis_unit(main_signal.get_axis_unit())
        
    def rename_signal(self, old_signal_name, new_signal_name):
        super().rename_signal(old_signal_name=old_signal_name, new_signal_name=new_signal_name)
        if self.main_signal_name == old_signal_name:
            self.set_main_signal(new_signal_name, update_axis=True)
        if old_signal_name in self.visible_signal_names:
            self.visible_signal_names[self.visible_signal_names.index(old_signal_name)] = new_signal_name
    
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
    
    def keys(self):
        return self.get_available_signals()
    
    def get_available_signals(self):
        return self.signals.keys()
    
    def __iter__(self):
        return self.signals.__iter__()
    
    def set_visible_signals(self, *signal_names):
        self.visible_signal_names = list(signal_names)
        self.set_main_signal(signal_names[0])
    
    def append_visible_signals(self, *signal_names):
        self.visible_signal_names.extend(list(signal_names))
        
    def remove_visible_signals(self, *signal_names):
        for signal_name in signal_names:
            self.visible_signal_names.remove(signal_name)
    
    def align_signal_axes(self, axis_limit) -> None:
        for signal in self.signals.keys():
            self.signals[signal].set_axis_limit(axis_limit)
        self.set_axis_limit(axis_limit)
    
    def set_real_axis_limit(self, axis_limit):
        self.align_signal_axes(axis_limit)
    
    def set_default_real_axis_limit(self):
        my_limit = None
        for sig_name in self:
            if my_limit is None:
                my_limit = self[sig_name].get_axis_limit()
            else:
                my_limit = (min(my_limit[0], self[sig_name].get_axis_limit()[0]), max(my_limit[1], self[sig_name].get_axis_limit()[1]))
        self.set_real_axis_limit(my_limit)
    
    def get_axis_margin(self):
        return self.axis_description.get_margin()
        
    def set_display_mode(self, display_mode):
        '''
        mode = 0: plot with main signal value label;
        mode = 1: plot with all value labels;
        mode = 2: plot separately;
        mode = 3: plot with denoted axis

        When mode is set to 1, axis_shift could be delivered to the plot function to adjust the position of the right axis
        '''
        self.display_mode = display_mode
        
    def plot_with_main_signal_value_label(self, **kwargs):
        
        ax: plt.Axes
        fig, ax = self.subplots(1, 1)
        axes = [ax]
        handles = []
        my_colormap = self.colormap
        my_colormap_min = self.colormap_min
        my_colormap_max = self.colormap_max
        need_legend = kwargs.pop("legend", True)
        legend_cols = kwargs.pop("legend_cols", 1)
        axis_digits = kwargs.pop("axis_digits", 1)
        value_digits = kwargs.pop("value_digits", 1)
        if my_colormap == "default":
            for i, signal_name in enumerate(self.visible_signal_names):
                signal = self.signals[signal_name]
                handles.append(signal.plot_at(ax, color=f"C{i}", **kwargs))
        else:
            if isinstance(my_colormap, list):
                if len(my_colormap) != len(self.visible_signal_names):
                    raise ValueError("The length of colormap should be the same as the number of visible signals")
                for i, signal_name in enumerate(self.visible_signal_names):
                    signal = self.signals[signal_name]
                    handles.append(signal.plot_at(ax, color=my_colormap[i], **kwargs)) 
            else: 
                my_len = len(self.visible_signal_names)
                if my_len > 1:
                    for i, signal_name in enumerate(self.visible_signal_names):
                        signal = self.signals[signal_name]
                        handles.append(signal.plot_at(ax, color=my_colormap(i / (my_len-1) * (my_colormap_max - my_colormap_min) + my_colormap_min), **kwargs))
                else:
                    signal = self.signals[self.visible_signal_names[0]]
                    handles.append(signal.plot_at(ax, color=my_colormap(0.6), **kwargs))
        main_signal = self[self.main_signal_name]
        xticks = main_signal.get_axis_ticks()
        xticklabels = main_signal.get_axis_ticklabels(digits=axis_digits)
        yticks = main_signal.get_value_ticks()
        yticklabels = main_signal.get_value_tick_labels(digits = value_digits)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(self.get_axis_label())
        ax.set_ylabel(main_signal.get_value_label())
        if need_legend:
            ax.legend(handles = handles, ncols=legend_cols)
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
            if i > 0:
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
    
    def plot_separately(self, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
        row_num = kwargs.pop("row", 0)
        col_num = kwargs.pop("col", 0)
        if row_num == 0 and col_num == 0:
            # 尽可能按照正方形进行 plot
            row_num = int(np.sqrt(len(self.visible_signal_names))) + 1 if np.sqrt(len(self.visible_signal_names)) % 1 != 0 else int(np.sqrt(len(self.visible_signal_names)))
            col_num = len(self.visible_signal_names) // row_num + 1 if len(self.visible_signal_names) % row_num != 0 else len(self.visible_signal_names) // row_num
        elif row_num == 0:
            # 按照 col_num 调整 row_num
            row_num = len(self.visible_signal_names) // col_num + 1 if len(self.visible_signal_names) % col_num != 0 else len(self.visible_signal_names) // col_num
        elif col_num == 0:
            # 按照 row_num 调整 col_num
            col_num = len(self.visible_signal_names) // row_num + 1 if len(self.visible_signal_names) % row_num != 0 else len(self.visible_signal_names) // row_num
            
        if row_num * col_num < len(self.visible_signal_names):
            raise ValueError(f"{row_num} rows and {col_num} columns are not enough to plot {len(self.visible_signal_names)} signals.")
        
        if row_num == 1 and col_num == 1:
            fig, ax = self.subplots(1, 1)
            axes = [[ax]]
        elif row_num == 1:
            fig, axes = self.subplots(1, col_num)
            axes = [axes]
        elif col_num == 1:
            fig, axes = self.subplots(row_num, 1)
            axes = [[ax] for ax in axes]
        else:
            fig, axes = self.subplots(row_num, col_num)
            
        for i, signal_name in enumerate(self.visible_signal_names):
            signal = self.signals[signal_name]
            row_index = i // col_num
            col_index = i % col_num
            signal.plot_at(axes[row_index][col_index], **kwargs)
            axes[row_index][col_index].set_title(signal_name)
            axes[row_index][col_index].set_xlabel(self.get_axis_label())
            axes[row_index][col_index].set_ylabel(signal.get_value_label())
            xticks, xticklabels = signal.get_axis_ticks(), signal.get_axis_ticklabels()
            axes[row_index][col_index].set_xticks(xticks)
            axes[row_index][col_index].set_xticklabels(xticklabels)
            yticks, yticklabels = signal.get_value_ticks(), signal.get_value_tick_labels()
            axes[row_index][col_index].set_yticks(yticks)
            axes[row_index][col_index].set_yticklabels(yticklabels)
            axes[row_index][col_index].set_xlim(0, 1)
            axes[row_index][col_index].set_ylim(0, 1)
        fig.tight_layout()
        return fig, axes
    
    def plot_with_denoted_axis(self, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:

        ax: plt.Axes
        fig, ax = self.subplots(1, 1)
        axes = [ax]
        handles = []
        my_colormap = kwargs.pop("cmap", None)
        need_legend = kwargs.pop("legend", True)
        legend_cols = kwargs.pop("legend_cols", 1)
        axis_digits = kwargs.pop("axis_digits", 1)
        value_digits = kwargs.pop("value_digits", 1)
        if not my_colormap:
            for i, signal_name in enumerate(self.visible_signal_names):
                signal = self.signals[signal_name]
                handles.append(signal.plot_at(ax, color=f"C{i}", **kwargs))
        else:
            my_len = len(self.visible_signal_names)
            if my_len > 1:
                for i, signal_name in enumerate(self.visible_signal_names):
                    signal = self.signals[signal_name]
                    handles.append(signal.plot_at(ax, color=my_colormap(i / (my_len-1) * 0.8 + 0.2), **kwargs))
            else:
                signal = self.signals[self.visible_signal_names[0]]
                handles.append(signal.plot_at(ax, color=my_colormap(0.6), **kwargs))
        xticks = self.get_axis_ticks()
        xticklabels = self.get_axis_ticklabels(digits=axis_digits)
        yticks = self.get_value_ticks()
        yticklabels = self.get_value_ticklabels(digits = value_digits)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(self.get_axis_label())
        ax.set_ylabel(self.get_value_label())
        if need_legend:
            ax.legend(handles = handles, ncols=legend_cols)
        ax.set_title(self.get_name())
        fig.tight_layout()
        return fig, axes
        
    def plot(self, **kwargs):
        '''
        mode = 0: plot with main signal value label;
        mode = 1: plot with all value labels;
        mode = 2: plot separately;
        mode = 3: plot with denoted axis
        legend_cols: int, default 1, number of columns in the legend
        '''
        axes: list[plt.Axes]
        if self.display_mode in [0, 1, 3]:
            # Axes containing only 1 subplot
            if self.display_mode == 0:
                fig, axes = self.plot_with_main_signal_value_label(**kwargs)
            elif self.display_mode == 1:
                axis_shift = kwargs.pop("axis_shift", 0.2)
                fig, axes = self.plot_with_all_value_labels(axis_shift=axis_shift, **kwargs)
            elif self.display_mode == 3:
                fig, axes = self.plot_with_denoted_axis(**kwargs)
            axes[0].set_title(self.get_name())
        elif self.display_mode in [2]:
            # Axes containing multiple subplots
            if self.display_mode == 2:
                fig, axes = self.plot_separately(**kwargs)
        else:
            raise Exception("Unknown display mode")
        return fig, axes

class ContinuousSignal1DCollection(Signal1DCollection):
    
    @classmethod
    def merge(cls, signal_collections: list['ContinuousSignal1DCollection'], name="Merged ContinuousSignal1DCollection"):
        signals = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals.values():
                signals.append(signal)
        return cls(signals, name=name)
    
    @classmethod
    def average(cls, signals: list[ContinuousSignal1D], name="Average ContinuousSignal1D"):
        '''
        Average the signals in the list and return a new Continuous
        '''
        if len(signals) == 0:
            raise ValueError("No signals to average")
        if len(signals) == 1:
            return signals[0]
        axis_name = signals[0].get_axis_name()
        axis_unit = signals[0].get_axis_unit()
        value_name = signals[0].get_value_name()
        value_unit = signals[0].get_value_unit()
        data: pd.DataFrame = signals[0].get_data().copy()
        data.iloc[:, 1] = 0
        for signal in signals:
            data.iloc[:, 1] += signal.get_data().iloc[:, 1]
        data.iloc[:, 1] /= len(signals)
        return cls(
            signals=[ContinuousSignal1D(data, name, axis_name, axis_unit, value_name, value_unit)],
            name=name
        )
    
    def __getitem__(self, key: str):
        res: ContinuousSignal1D = super().__getitem__(key)
        return res
