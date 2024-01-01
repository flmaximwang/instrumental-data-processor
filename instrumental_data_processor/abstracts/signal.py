import os
import re
import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import instrumental_data_processor.utils.path_utils as path_utils


class DescriptionAnnotation:
    """
    Every signal contains multiple descriptions.
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

    def set_limit(self, limit):
        pass

    def get_limit(self):
        pass 
    
    def set_tick_number(self, tick_number):
        pass

    def get_tick_number(self):
        pass

    def get_ticks(self):
        pass

    def get_tick_labels(self):
        pass
class NumericDescriptionAnnotation(DescriptionAnnotation):
    
    def __init__(self, name: str, unit: str, limit: tuple[int | float, int | float], tick_number: int):
        super().__init__(name, unit)
        self.limit: tuple[int | float, int | float] = (0, 0)
        self.set_limit(limit)
        self.tick_number: int = 0
        self.set_tick_number(tick_number)

    def set_limit(self, limit: tuple[int | float, int | float]):
        if not isinstance(limit, tuple):
            raise TypeError("limit must be a tuple")
        if len(limit) != 2:
            raise ValueError("limit must be a tuple with length 2")
        if not isinstance(limit[0], (int, float)) or not isinstance(limit[1], (int, float)):
            raise TypeError("limit must be a tuple with two numbers")
        self.limit = limit

    def get_limit(self):
        return self.limit
    
    def set_tick_number(self, tick_number):
        self.tick_number = tick_number

    def get_tick_number(self):
        return self.tick_number

    def get_ticks(self):
        return np.linspace(0, 1, self.get_tick_number())

    def get_tick_labels(self, digits=1):
        return np.round(np.linspace(self.get_limit()[0], self.get_limit()[1], self.get_tick_number()), digits)
        
class AnnotationDescriptionAnnotation(DescriptionAnnotation):

    def get_limit(self):
        return (0, 1)

    def get_tick_number(self):
        return 0

    def get_ticks(self):
        return [0]

    def get_tick_labels(self):
        return [""]

class Signal:
    """
    a signal is composed of multiple descriptions. These descriptions are represented
    by columns in a pandas.DataFrame and therefore have equal lengths.
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
        self.description_annotations: list[DescriptionAnnotation] = []

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
    ) -> DescriptionAnnotation:
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
