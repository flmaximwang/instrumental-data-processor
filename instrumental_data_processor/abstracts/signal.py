import os
import pandas as pd
import matplotlib.pyplot as plt
import instrumental_data_processor.utils.path_utils as path_utils

class Signal:
    
    @classmethod
    def from_csv(cls, path, name=None, **kwargs):
        '''
        See pandas.read_csv for supported kwargs
        '''
        if name is None:
            name = path_utils.get_name_from_path(path)
        data = pd.read_csv(path, **kwargs)
        return cls(data, name)
    
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
        
    def set_name(self, name):
        self.name = name
        
    def get_name(self):
        return self.name
    
    def set_data(self, data):
        self.data = data