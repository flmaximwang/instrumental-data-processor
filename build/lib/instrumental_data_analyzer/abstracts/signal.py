import os
import re
import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import instrumental_data_analyzer.utils.path_utils as path_utils
from ..utils import transform_utils

class DescAnno:
    """
    Every signal contains multiple descriptions, which is continuous / discrete.
    This class is used to annotate each description for its
    type (like Volume), unit (like mL), limit (like (0, 10)) and ticks (like tick_number = 5)
    """

    def __init__(self, name: str, unit: str):
        self.name = name
        self.unit = unit

    def set_name(self, type):
        self.name = type

    def get_name(self):
        return self.name

    def set_unit(self, unit):
        self.unit = unit

    def get_unit(self):
        return self.unit
    
    def get_label(self):
        if self.get_unit() is None:
            return self.get_name()
        else:
            return f"{self.get_name()} ({self.get_unit()})"
    
    def get_limit(self):
        pass

    def set_limit(self, limit: tuple[int | float, int | float]):
        pass
        
    def get_margin(self):
        pass
    
    def set_margin(self, margin: tuple[int | float, int | float]):
        pass

    def set_tick_number(self, tick_number):
        pass

    def get_tick_number(self):
        pass

    def get_ticks(self):
        pass

    def get_ticklabels(self):
        pass

    def copy(self):
        return self.__class__(self.get_name(), self.get_unit())
class ContDescAnno(DescAnno):
    
    def __init__(
        self,
        name: str,
        unit: str,
        limit: tuple[float, float] = (0, 0),
        tick_number: int = 6,
        margin: tuple[float, float] = (0, 0),
        digits = 1,
    ):
        super().__init__(name, unit)
        self.limit: tuple[int | float, int | float] = (0, 0)
        self.margin: tuple[int | float, int | float] = (0, 0)
        self.set_limit(limit)
        self.tick_number: int = 0
        self.set_tick_number(tick_number)
        self.margin = (0, 0)
        self.set_margin(margin)
        self.digits = digits
    
    def get_limit(self):
        return self.limit

    def set_limit(self, limit: tuple[int | float, int | float]):
        if not isinstance(limit, tuple):
            raise TypeError("limit must be a tuple")
        if len(limit) != 2:
            raise ValueError("limit must be a tuple with length 2")
        if not isinstance(limit[0], (int, float)) or not isinstance(limit[1], (int, float)):
            raise TypeError("limit must be a tuple with two numbers")
        self.limit = limit
        
    def get_margin(self):
        return self.margin
    
    def set_margin(self, margin: tuple[int | float, int | float]):
        if not isinstance(margin, tuple):
            raise TypeError("margin must be a tuple")
        if len(margin) != 2:
            raise ValueError("margin must be a tuple with length 2")
        if not isinstance(margin[0], (int, float)) or not isinstance(margin[1], (int, float)):
            raise TypeError("margin must be a tuple with two numbers")
        self.margin = margin
        
    def get_digits(self):
        return self.digits
    
    def set_digits(self, digits):
        self.digits = digits
    
    def set_tick_number(self, tick_number):
        self.tick_number = tick_number

    def get_tick_number(self):
        return self.tick_number

    def get_ticks(self):
        """
        Generate a sequence of ticks within specified margins.

        Parameters:
        lower_margin (float): The lower margin as a fraction of the total range. Default is 0.
        higher_margin (float): The higher margin as a fraction of the total range. Default is 0.

        Returns:
        numpy.ndarray: An array of tick values linearly spaced between the calculated lower and higher limits.
        """
        lower_margin, higher_margin = self.get_margin()
        lower_limit = lower_margin / (1 + lower_margin + higher_margin)
        higher_limit = 1 - higher_margin / (1 + lower_margin + higher_margin)
        return np.linspace(lower_limit, higher_limit, self.get_tick_number())

    def get_ticklabels(self):
        def _get_ticklabels():
            return np.around(
                np.linspace(
                    self.get_limit()[0],
                    self.get_limit()[1],
                    self.get_tick_number()
                ),
                decimals=self.digits
            )
        tick_labels = _get_ticklabels()
        # 检查 tick_labels 中是否有重复的项; 如果有, 提高 digits, 直到没有重复
        while len(tick_labels) != len(set(tick_labels)):
            self.digits += 1
            tick_labels = _get_ticklabels()
        return tick_labels
    
    def copy(self):
        return self.__class__(self.get_name(), self.get_unit(), self.get_limit(), self.get_tick_number())
        
class DiscDescAnno(DescAnno):

    def get_limit(self):
        return (0, 1)

    def get_tick_number(self):
        return 0

    def get_ticks(self):
        return [0]

    def get_ticklabels(self):
        return [""]
    
    def copy(self):
        return self.__class__(self.get_name(), self.get_unit())

class Signal:
    """
    a signal is composed of multiple descriptions.
    At least one description is continuous for all instruments.
    These descriptions are represented by columns in a pandas.DataFrame and therefore have equal lengths.
    More specific signals are subclasses that have defined descriptions.
    """

    @staticmethod
    def get_type_and_unit_from_header(header):
        """
        Returns type and unit from header string
        """
        type_unit_pattern = re.match(r"(.*) \((.*)\)", header)
        if type_unit_pattern is None:
            raise ValueError(
                'Expected header to be in format "type (unit)", but got {}'.format(
                    header
                )
            )
        return type_unit_pattern.group(1), type_unit_pattern.group(2)

    @classmethod
    def from_csv(cls, path, name=None, **kwargs):
        """
        See pandas.read_csv for supported kwargs
        """
        if name is None:
            name = path_utils.get_name_from_path(path)
        data = pd.read_csv(path, **kwargs)
        return cls(data, name)

    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
        # axis_annotations are annotations that describe each axis
        self.description_annotations: list[DescAnno] = []

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def get_data(self):
        return self.data

    def get_description_by_index(self, description_index: int) -> pd.Series:
        return self.data.iloc[:, description_index]

    def get_description_annotations_by_index(
        self, description_index: str
    ) -> DescAnno:
        return self.description_annotations[description_index]

    def get_limit(self, description_index):
        return self.get_description_annotations_by_index(description_index).get_limit()

    def set_limit(
        self, description_index, limit: tuple[int | float, int | float] | None = None
    ):
        if limit is None:
            self.get_description_annotations_by_index(description_index).set_limit(
                (
                    self.get_description_by_index(description_index).min(),
                    self.get_description_by_index(description_index).max(),
                )
            )
        else:
            self.get_description_annotations_by_index(description_index).set_limit(limit)
    
    def get_margin(self, description_index):
        return self.get_description_annotations_by_index(description_index).get_margin()
    
    def set_margin(
        self, 
        description_index, 
        margin: tuple[int | float, int | float]
    ):
        self.get_description_annotations_by_index(description_index).set_margin(margin)
        
    @staticmethod
    def rescale_for_plots(series, limit, margin):
        limit_range = limit[1] - limit[0]
        return transform_utils.rescale_to_0_1(
            series,
            limit[0] - limit_range * margin[0],
            limit[1] + limit_range * margin[1],
        )

    def export(self, export_path, mode="write"):
        directory = os.path.dirname(export_path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        if os.path.exists(export_path):
            if mode == "write":
                raise Exception(f"File {export_path} already exists")
            elif mode == "replace":
                os.remove(export_path)
            else:
                raise ValueError("mode should be either 'write' or 'replace'")
        self.get_data().to_csv(export_path)
