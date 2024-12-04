import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections.abc import Iterable

def extract_number_from_chrom(chrom_string):
    '''
    从chromatogram的字符串中提取编号
    '''
    my_pattern = re.compile(r'Chrom\.(\d+)')
    my_match = my_pattern.match(chrom_string)
    
    if my_match:
        return int(my_match.group(1)) - 1
    else:
        return None

def rescale_signal(signal_list, target_min, target_max):
    '''
    target_min 会被缩放到 0, target_max 会被缩放到 1
    signal_data 会根据以上的规则线性缩放
    '''
    return (signal_list - target_min) / (target_max - target_min)

class UnicornData:
    
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(filename, sep='\t', encoding='utf-16', header=None)
        # 第一行显示 chromatogram 的编号, 第二行显示信号的类型, 第三行为交替的时间和信号值
        results = {}
        for n, chrom_string in enumerate(self.data.iloc[0, 0::2]):
            chrom_number = extract_number_from_chrom(chrom_string)
            if not chrom_number in results.keys():
                results[chrom_number] = []
                results[chrom_number].append(n)
            else:
                results[chrom_number].append(n)
        self.chromatograms = {}
        for i in results.keys():
            self.chromatograms[i] = UnicornChromtogram(self.data, results[i])
            
        print('Chromatograms: {}'.format(len(self.chromatograms.keys())))
        print("Use 'data.chromatograms[0].signals' to get signals of chromatogram 0")
        print("Use 'data.chromatograms[0].UV' or 'data.chromatogram[0].__getattribute__(\"UV\")' to get the UV signal of chromatogram 0")
        
    def get_chromatograms(self):
        return self.chromatograms

class UnicornChromtogram:
    
    def __init__(self, data, col_indices):
        self.signals = []
        for col_index in col_indices:
            signal = data.iloc[1, 2*col_index]
            self.signals.append(signal)
            signal_data:pd.DataFrame = data.iloc[3:, 2*col_index:2*col_index+2]
            signal_data.columns = data.iloc[2, 2*col_index:2*col_index+2]
            signal_data = signal_data.dropna(subset=[signal_data.columns[0]])
            # print(signal)
            if not signal in ["Run Log", "Fraction", "Injection"]:
                signal_data = signal_data.astype(float)
            else:
                # 仅仅设置体积列为 float
                signal_data.iloc[:, 0] = signal_data.iloc[:, 0].astype(float)
            self.__setattr__(signal, signal_data)
        signals_to_extend = []
        for signal in self.signals:
            # print(signal)
            if not signal in ["Run Log", "Fraction", "Injection"]:
                signals_to_extend.append(signal + "_normalized")
                signal_data_normalized = self.__getattribute__(signal).copy()
                signal_data_normalized.iloc[:, 1] = signal_data_normalized.iloc[:, 1] / signal_data_normalized.iloc[:, 1].max()
                self.__setattr__(signal + "_normalized", signal_data_normalized)
        self.signals.extend(signals_to_extend)
        
        self.visible_signals = ["UV", "Cond"]
        self.xlimit = []
        self.set_default_x_limit()
        self.y_limits = {}
        for signal in self.signals:
            self.set_default_y_limit_for_signal(signal)
        self.main_signal = "UV"
        self.xlabel = "Volume (ml)"
        self.x_tick_number = 10
        self.y_tick_number = 10
        self.plot_fraction_flag = False
        self.figsize = None
        
    def set_default_x_limit(self):
        my_min = 0
        my_max = 0
        for signal in self.signals:
            if signal in ["Run Log", "Fraction"]:
                continue
            if (len(self.__getattribute__(signal).iloc[:, 0]) == 0):
                continue
            my_min = min(min(self.__getattribute__(signal).iloc[:, 0]), my_min)
            my_max = max(max(self.__getattribute__(signal).iloc[:, 0]), my_max)
        self.x_limit = [my_min, my_max]
    
    def set_x_limit(self, my_min, my_max):
        self.x_limit = [my_min, my_max]
    
    def set_default_y_limit_for_signal(self, signal):
        self.y_limits[signal] = [0, 0]
        for signal in self.signals:
            if (len(self.__getattribute__(signal).iloc[:, 1]) == 0):
                continue
            self.y_limits[signal] = [min(self.__getattribute__(signal).iloc[:, 1]), max(self.__getattribute__(signal).iloc[:, 1])]
            
    def set_y_limit_for_signal(self, signal, my_min, my_max):
        self.y_limits[signal] = [my_min, my_max]

    def set_figsize(self, figsize_width, figsize_height):
        self.figsize = (figsize_width, figsize_height)
        
    def unset_figsize(self):
        self.figsize = None
    
    def set_visible_signals(self, *signals):
        self.visible_signals = signals
        
    def set_main_signal(self, signal):
        self.main_signal = signal
    
    def set_x_ticks(self):
        tick_labels = np.linspace(self.x_limit[0], self.x_limit[1], self.x_tick_number)
        tick_labels = np.round(tick_labels, 1)
        plt.xticks(np.linspace(self.x_limit[0], self.x_limit[1], self.x_tick_number), tick_labels)
        
    def set_y_ticks(self):
        signal = self.main_signal
        tick_labels = np.linspace(self.y_limits[signal][0], self.y_limits[signal][1], self.y_tick_number)
        tick_labels = np.round(tick_labels, 1)
        plt.yticks(np.linspace(0, 1, self.y_tick_number), tick_labels)
    
    def set_labels(self):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.__getattribute__(self.main_signal).columns[1])
    
    def plot_signals(self, fraction=False, fraction_signal_height=0.2, fraction_font_size=5, title=None):
        if self.figsize:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig, ax = plt.subplots(1, 1)
        for signal in self.visible_signals:
            ax.plot(self.__getattribute__(signal).iloc[:, 0], rescale_signal(self.__getattribute__(signal).iloc[:, 1], self.y_limits[signal][0], self.y_limits[signal][1]))
        if fraction:
            self.plot_fractions(fraction_signal_height=fraction_signal_height, fraction_font_size=fraction_font_size)
        self.set_x_ticks()
        self.set_y_ticks()
        self.set_labels()
        ax.set_xlim(self.x_limit)
        ax.set_ylim([0, 1])
        ax.legend(self.visible_signals)
        ax.set_title(title)
        
        return fig, ax
        
    def plot_fractions(self, fraction_signal_height=0.2, fraction_font_size=5):
        for index, fraction in self.get_signal("Fraction").iterrows():
            plt.vlines(fraction[0], 0, fraction_signal_height, colors='red', linestyles='dashed', linewidths = 0.5)
            plt.annotate(fraction[1], (fraction[0], fraction_signal_height), fontsize=fraction_font_size, rotation=90)
    
    def get_signal(self, signal):
        if not signal in self.signals:
            print("No such signal: {}".format(signal))
            print("Available signals: {}".format(self.signals))
            return None
        return self.__getattribute__(signal).copy()
    
    def get_signal_between(self, signal, start, end):
        if not signal in self.signals:
            print("No such signal: {}".format(signal))
            print("Available signals: {}".format(self.signals))
            return None
        signal_data = self.__getattribute__(signal).copy()
        signal_data = signal_data[(signal_data.iloc[:, 0] >= start) & (signal_data.iloc[:, 0] <= end)]
        return signal_data
       
    def integrate_signal_between(self, signal:str, start, end, ax:plt.Axes=None, color='red', alpha=0.1, linestyles='dashed', linewidths=0.5, baseline=0.0):
        '''
        start 和 end 为体积
        '''
        
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
