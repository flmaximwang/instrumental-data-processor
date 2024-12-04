from ..abstracts import Matrix
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MultiWellPlate(Matrix):
    
    OMNI_EASY_CALIBRATION = [0, 125, 250, 500, 750, 1000, 1500, 2000]
    
    @staticmethod
    def from_csv(filepath: str, header = None, index_col = None, sep=","):
        return MultiWellPlate(pd.read_csv(filepath, header=header, index_col=index_col, sep=sep))
    
    def __init__(self, data: pd.DataFrame, name: str = "Default Multi-Well Plate"):
        super().__init__(data, name)
        # Check the type of the multi-well plate
        # The plate should be 6-well (3x2), 12-well (4x3), 24-well (6x4), 48-well (8x6), 96-well (8x12) or 384-well (16x24)
        
        if self.data.shape == (3, 2):
            self.type = '6-well'
            self.data.columns = [1, 2, 3]
            self.data.index = ['A', 'B']
        elif self.data.shape == (4, 3):
            self.type = '12-well'
            self.data.columns = [1, 2, 3, 4]
            self.data.index = ['A', 'B', 'C']
        elif self.data.shape == (6, 4):
            self.type = '24-well'
            self.data.columns = [1, 2, 3, 4, 5, 6]
            self.data.index = ['A', 'B', 'C', 'D']
        elif self.data.shape == (8, 6):
            self.type = '48-well'
            self.data.columns = [1, 2, 3, 4, 5, 6, 7, 8]
            self.data.index = ['A', 'B', 'C', 'D', 'E', 'F']
        elif self.data.shape == (8, 12):
            self.type = '96-well'
            self.data.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            self.data.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        elif self.data.shape == (16, 24):
            self.type = '384-well'
            self.data.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            self.data.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
            
    def __getitem__(self, key: tuple[str, int]):
        return self.data.loc[key[0], key[1]]
    
    def calibration_and_measurement(self,
            calibration_indices: list[float],
            calibration_markers: list[tuple[str, int]],
            measurement_markers: list[tuple[str, int]],
            xlabel: str,
            ylabel: str,
            calibration_scatter_color = 'black',
            calibration_scatter_size = 30,
            calibration_scatter_marker = '+',
            calibration_line_color = 'red',
            calibration_line_width = 1,
            measurement_scatter_color: list[str] | str = 'default',
            measurement_scatter_size = 30,
            measurement_scatter_marker = '^',
            legend_col_num = 1
        ):

        if len(calibration_indices) != len(calibration_markers):
            raise ValueError('The length of calibration_index and calibration_values should be the same')
        x = pd.Series(calibration_indices)
        y = []
        for i in calibration_markers:
            y.append(self[i[0], i[1]])
        y = pd.Series(y)
        result = st.linregress(x, y)
        y_fitting = x.apply(lambda x:result.slope*x + result.intercept)
        
        fig, ax = plt.subplots(1, 1)
        ax.scatter(x, y, s=calibration_scatter_size, c=calibration_scatter_color, marker=calibration_scatter_marker)
        ax.plot(x, y_fitting, c=calibration_line_color, linewidth=calibration_line_width)
        
        measurement_values = list(map(lambda x: self[x[0], x[1]], measurement_markers))
        measurement_indices = pd.Series(map(lambda x: (x - result.intercept) / result.slope, measurement_values))
        if measurement_scatter_color == 'default':
            for i, [mindex, mvalue] in enumerate(zip(measurement_indices, measurement_values)):
                ax.scatter(mindex, mvalue, s=measurement_scatter_size, c=f"C{i}", marker=measurement_scatter_marker,
                        label=f"{measurement_markers[i][0]}{measurement_markers[i][1]}")
        elif isinstance(measurement_scatter_color, str):
            ax.scatter(measurement_indices, measurement_values, s=measurement_scatter_size, c=measurement_scatter_color, marker=measurement_scatter_marker)
        else:
            if len(measurement_indices) != len(measurement_scatter_color):
                raise ValueError('The length of measurement_indices and measurement_scatter_color should be the same')
            for i, [mindex, mvalue, mcolor] in enumerate(zip(measurement_indices, measurement_values, measurement_scatter_color)):
                ax.scatter(mindex, mvalue, s=measurement_scatter_size, c=mcolor, marker=measurement_scatter_marker,
                        label=f"{measurement_markers[i][0]}{measurement_markers[i][1]}")
            ax.legend(ncol = legend_col_num)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        result_table = pd.DataFrame(columns=['Markers', 'Indices', 'Sigmas'])
        result_table['Markers'] = [i[0] + str(i[1]) for i in measurement_markers]
        result_table['Indices'] = measurement_indices
        result_table['Sigmas'] = np.sqrt((result.intercept_stderr / result.slope)**2 + (measurement_values - result.intercept)**2 / result.slope ** 4 * result.stderr**2 )
        print(result_table.to_markdown())

        return fig, ax, result, result_table