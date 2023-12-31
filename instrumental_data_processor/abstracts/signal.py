import os
import pandas as pd
import matplotlib.pyplot as plt
import path_utils

class Signal:
    
    @classmethod
    def from_csv(cls, path, name=None):
        if name is None:
            name = ".".join(os.path.basename(path).split('.')[:-1])
        data = pd.read_csv(path)
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