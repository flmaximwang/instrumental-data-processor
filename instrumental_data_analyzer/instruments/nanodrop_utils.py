import os
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as st
from scipy.optimize import curve_fit

class UnitConverter:

    @staticmethod
    def inch_to_cm(inch):
        return inch * 2.54

    @staticmethod
    def cm_to_inch(cm):
        return cm / 2.54

def set_theme(ax:plt.Axes, xlim=(200, 800), ylim=(0, 1), xlabel="Wavelength (nm)", ylabel="1 mm Absorbance", title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

class Spectrum:
    
    def __init__(self, input, name="", comment=""):
        '''
        input: a pandas DataFrame with 2 columns, the first column is wavelength, the second column is intensity, 
            or a path to a csv file
        '''
        if isinstance(input, str):
            input_spectrum = pd.read_csv(input, skiprows=1) # read csv file
            with open(input, "r") as file_input:
                self.comment = file_input.readline()
            self.name = input.replace(".csv", "")
        elif isinstance(input, pd.DataFrame):
            input_spectrum = input
            self.name = name
            self.comment = comment
        else:
            raise TypeError("Unknown input type! Should be a pandas DataFrame or a path to a csv file. type(input) = " + str(type(input)))
            
        if len(input_spectrum.columns) != 2:
            raise ValueError("A spectrum should have 2 columns! The input has " + str(len(input_spectrum.columns)) + " columns. Check the inpu!")
        self.spectrum = input_spectrum
        
    def export(self, default_directory = "./assets"):
        comment = self.comment
        export_path = os.path.join(default_directory, self.name + ".csv")
        with open(export_path, "w") as file_output:
            file_output.write(comment + "\n")
        self.spectrum.to_csv(export_path, sep=",", index=False, mode="a")
    
    def copy(self):
        return Spectrum(self.spectrum.copy(), self.name, self.comment)
    
    def get_name(self):
        return self.name
    
    def rename(self, new_name):
        self.name = new_name
        
    def get_spectrum(self):
        return self.spectrum
    
    def get_intensity_at(self, wavelength, ax:plt.Axes=None, color="red", linestyle="--", text_shift=(0, 0.05)):
        '''
        返回某一波长的强度
        给定一个 ax, 可以在 ax 上标注该波长的强度
        '''
        data_index = self.get_spectrum().index[self.spectrum.iloc[:, 0] == wavelength]
        intensity = self.get_spectrum().iloc[data_index, 1]
        if ax:
            ax.vlines(wavelength, 0, intensity, color = color, linestyle = linestyle)
            ax.annotate(" ".join([str(wavelength), "nm"]), (wavelength + text_shift[0], intensity + (ax.get_ylim()[1] - ax.get_ylim()[0]) * text_shift[1]), color = color)
        return intensity
    
    def get_peak_between(self, min_wavelength, max_wavelength, ax: plt.Axes=None, color="red", linestyle="--", text_shift=(0, 0.05)):
        '''
        返回某一波长范围内的峰值波长以及峰值强度
        给定一个ax，可以在ax上绘制峰值
        '''
        data_index = self.spectrum.index[(self.spectrum["Wavelength (nm)"] > min_wavelength) & (self.spectrum["Wavelength (nm)"] < max_wavelength)]
        # print(data_index)
        data = self.spectrum.iloc[data_index, :]
        max_index = data.iloc[:, 1].idxmax()
        max_x = self.spectrum.iloc[max_index, 0]
        max_intensity = self.spectrum.iloc[max_index, 1]
        if ax:
            ax.vlines(max_x, 0, max_intensity, color = color, linestyle = linestyle)
            ax.annotate(" ".join([str(max_x), "nm"]), (max_x + text_shift[0], max_intensity + (ax.get_ylim()[1] - ax.get_ylim()[0]) * text_shift[1]), color = color)
        return max_x, max_intensity
    
    def get_extinction_coefficient_at(self, wavelength, concentration, molecular_weight=None, path_length=0.1):
        '''
        concentration should be in mg/ml or mM
        path length: in cm
        molecular weight: in Da; if no molecular weight is given, the concentration is assumed to be in mM, otherwise in mg/ml
        '''
        intensity = self.read_intensity_at(wavelength)
        if not molecular_weight: # concentration in mM
            return intensity / concentration / path_length
        else: # concentration in mg/ml
            return intensity / concentration * molecular_weight / path_length / 1000
    
    def plot_spectrum_at(self, ax: plt.Axes, color="black", label=None):
        ax.plot(self.spectrum.iloc[:, 0], self.spectrum.iloc[:, 1], color=color, label=label)
        
    def normalize_at(self, wavelength, value=1):
        '''
        normalize the intensity at a given wavelength to a given value
        '''
        intensity = self.get_intensity_at(wavelength).item()
        spectrum = self.get_spectrum()
        spectrum.iloc[:, 1] = spectrum.iloc[:, 1] / intensity * value

class Workbook:
    
    def __init__(self, tsv_path):
        self.spectrums = []
        with open(tsv_path) as file_input:
            flag = "start"
            entryName = ""
            time = ""
            csv_text = ""
            
            for line in file_input:
                # print(flag, line)
                if flag == "start":
                    if not re.match(r"^\s*$", line):
                        entryName = line[:-1]
                        flag = "time"
                    else:
                        continue # skip empty lines
                elif flag == "time":
                    time = line[:-1]
                    flag = "content"
                elif flag == "content":
                    if line.startswith("\n"):
                        flag = "start"
                        self.spectrums.append(Spectrum(pd.read_csv(io.StringIO(csv_text)), entryName, time))
                        csv_text = ""
                    else:
                        csv_text += line.replace("\t", ",")
                else:
                    raise ValueError("Unknown flag: " + flag)
            
            # append the last one
            if flag == "content":
                self.spectrums.append(Spectrum(pd.read_csv(io.StringIO(csv_text)), entryName, time))
    
    def get_spectrums(self) -> list[Spectrum]:
        return self.spectrums
    
    def set_spectrum(self, spectrum: pd.DataFrame):
        self.spectrum = spectrum

class BatchPlot:

    @staticmethod
    def draw_spectrums_separately(input_list: list[pd.DataFrame] or list[str], layout: tuple[int, int], figsize: tuple[float, float], xlim: tuple[float, float], ylim: tuple[float, float], xlabel = True, ylabel = True):
        '''
        分别绘制所有的光谱
        '''
        fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
        res_axes = []
        for num, input_data in enumerate(input_list):
            table = read_spectrum(input_data)
            x_name = table.columns[0]
            y_name = table.columns[1]
            x = table[x_name]
            y = table[y_name]
            if (layout[0] == 1) & (layout[1] == 1):
                c_ax = axes
            elif (layout[0] == 1) & (layout[1] > 1):
                c_ax = axes[num]
            elif (layout[0] > 1) & (layout[1] == 1):
                c_ax = axes[num]
            else:
                c_ax = axes[num%layout[0], num//layout[0]]
            res_axes.append(c_ax)
            c_ax.plot(x, y)
            c_ax.set_xlim(xlim[0], xlim[1])
            c_ax.set_ylim(ylim[0], ylim[1])
            c_ax.set_title(input_data)
        fig.tight_layout()
        return fig, res_axes

    @staticmethod
    def draw_spectrums_together(input_list: list[pd.DataFrame] or list[str], figsize, xlim, ylim, pallette = cm.get_cmap("viridis"), xlabel = True, ylabel = True, label_list = []):
        '''
        将所有的光谱绘制在一张图上
        '''
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i, input_data in enumerate(input_list):
            table = read_spectrum(input_data)
            x_name = table.columns[0]
            y_name = table.columns[1]
            x = table[x_name]
            y = table[y_name]
            if label_list:
                ax.plot(x, y, label = label_list[i], color = pallette(i / len(input_list) * 0.8 + 0.2))
            else:
                ax.plot(x, y, label = input_data.replace(".tsv", ""), color = pallette(i / len(input_list) * 0.8 + 0.2)) 
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
        return fig, ax

    @staticmethod
    def draw_wavelength_vs_groups(figsize: tuple[float, float], layout: tuple[float, float], wavelength: int or list, x_label: str, input_table: pd.DataFrame, ylim = (0, 1)):
        '''
        比较某一波长在不同组别间的差异
        将每列绘制在同一张图中
        横坐标为行
        '''
        row = layout[0]
        col = layout[1]
        fig, axes = plt.subplots(row, col, figsize=figsize)
        results = pd.DataFrame()
        xs = [int(i) for i in input_table.index]
        counter = -1
        for group in input_table:
            ys = []
            counter += 1
            for index in input_table.index:
                input_data = pd.read_csv(input_table.loc[index, group])
                if isinstance(wavelength, int):
                    data_index = input_data["Wavelength (nm)"] == wavelength
                else:
                    data_index = input_data["Wavelength (nm)"] == wavelength[counter]
                ys.append(float(input_data.loc[data_index, "1mm Absorbance"]))
            if (row == 1) & (col == 1):
                c_ax = axes
            elif (row == 1) & (col > 1):
                c_ax = axes[counter]
            elif (row > 1) & (col == 1):
                c_ax = axes[counter]
            else:
                c_ax = axes[counter//row, counter%row]
            c_ax.scatter(xs, ys, color = "black", marker = "+")
            result = st.linregress(xs, ys)
            results[group] = [result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr]
            y_fitted = pd.Series(xs).apply(lambda x: result.slope * x + result.intercept)
            c_ax.plot(xs, y_fitted, color = "red", linestyle = "--")
            c_ax.set_ylim(ylim)
            c_ax.set_xlabel(x_label)
            c_ax.set_ylabel("Absorbance")
            c_ax.set_title(group + ": " + str(wavelength[counter]) + " " + "nm")
        results.index = ["slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
        return fig, axes, results
    
