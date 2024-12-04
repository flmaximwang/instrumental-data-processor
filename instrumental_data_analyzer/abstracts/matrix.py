'''
Matrices are 2D arrays of values
'''

import pandas as pd

class Matrix:
    
    def __init__(self, data: pd.DataFrame, name: str = "Default Matrix"):
        if len(data.shape) != 2:
            raise ValueError("Matrix should be a 2D array")
        self.data = data
        self.name = name
    
    def get_data(self):
        return self.data
    
    def get_name(self):
        return self.name