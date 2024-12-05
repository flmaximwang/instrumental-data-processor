from ..concretes.voltammetry import *
from ..concretes.imp_spec import *
from ..utils.name_utils import rename_duplicated_names
import pandas as pd
import io
import os

class chi660eEIS():
    
    @staticmethod
    def txt_to_csv(txt_path, csv_name):
        csv_txt = ""
        with open(txt_path, 'r') as f:
            for i, line in enumerate(f):
                if i != 1:
                    csv_txt += line
        # Read csv_txt
        my_df = pd.read_csv(io.StringIO(csv_txt))
        my_df.drop(columns=[' Z/ohm', " Phase/deg"], inplace=True)
        file_name = txt_path.replace('.txt', '.csv') if not csv_name else os.path.join(os.path.dirname(txt_path), csv_name + ".csv")
        my_df.to_csv(file_name, index=False, header=False)
        
    @staticmethod
    def from_exported_files(txt_paths, name="Default Impedance Spectrum Collection"):
        '''
        txt_paths: list of path (str) or dict of path (str): label (str)
        '''
        my_data_list = []
        if isinstance(txt_paths, list):
            path_list = txt_paths,
            name_list = [os.path.basename(txt_path) for txt_path in txt_paths]
        elif isinstance(txt_paths, dict):
            path_list = list(txt_paths.keys())
            name_list = list(txt_paths.values())
        # 将 name_list 中重复的名字添加后缀
        
        rename_duplicated_names(name_list)
        
        for path, sig_name in zip(path_list, name_list):
            data = ""
            with open(path, 'r') as f:
                flag = False
                for line in f:
                    if not flag and "Freq/Hz" in line:
                        flag = True
                    if flag:
                        data += line
            my_df = pd.read_csv(io.StringIO(data))
            my_df.rename(columns={"Freq/Hz": "Frequency (Hz)", " Z'/ohm": "Z' (ohm)", " Z\"/ohm": "Z\" (ohm)"}, inplace=True)
            my_df = my_df[["Frequency (Hz)", "Z' (ohm)", "Z\" (ohm)"]]
            my_data_list.append(ImpedanceSpectrum(data = my_df, name=sig_name))
            
        return ImpedanceSpectrumCollection(signals=my_data_list, name=name)
        

class chi660eCV(VoltammegramCollection):
    
    @staticmethod
    def from_exported_files(txt_paths, name="CV"):
        '''
        txt_paths: list of path (str) or dict of path (str): label (str)
        '''
        my_data_list = []
        if isinstance(txt_paths, list):
            path_list = txt_paths
            name_list = [os.path.basename(txt_path) for txt_path in txt_paths]
        elif isinstance(txt_paths, dict):
            path_list = list(txt_paths.keys())
            name_list = list(txt_paths.values())
        # 将 name_list 中重复的名字添加后缀
        # print(txt_paths)
        # print(path_list)
        # print(name_list)
        
        rename_duplicated_names(name_list)
            
        for path, sig_name in zip(path_list, name_list):
            data = ""
            with open(path, 'r') as f:
                flag = False
                for line in f:
                    if not flag and "Potential/V" in line:
                        flag = True
                    if flag:
                        data += line
            my_df = pd.read_csv(io.StringIO(data))
            my_df.rename(columns={"Potential/V": "Potential (V)", " Current/A": "Current (A)"}, inplace=True)
            E_init = my_df.loc[0, "Potential (V)"]
            E_max = max(my_df["Potential (V)"])
            E_min = min(my_df["Potential (V)"])
            scan_direction = "Positive" if my_df.loc[1, "Potential (V)"] > E_init else "Negative"
            my_df["Time"] = my_df.index.copy()
            my_df["Segment"] = 1
            # 将连续递增/递减的部分分为一个段 segment
            if len(my_df) <= 2:
                pass
            else:
                for i in range(1, len(my_df) - 1):
                    if (my_df.loc[i, "Potential (V)"] - my_df.loc[i-1, "Potential (V)"]) * (my_df.loc[i+1, "Potential (V)"] - my_df.loc[i, "Potential (V)"]) < 0:
                        my_df.loc[i:, "Segment"] = my_df.loc[i-1, "Segment"] + 1
                    else:
                        my_df.loc[i, "Segment"] = my_df.loc[i-1, "Segment"]
                my_df.loc[len(my_df) - 1, "Segment"] = my_df.loc[len(my_df) - 2, "Segment"]
            
            # 将 Time 放到第四列
            cols = ["Potential (V)", "Current (A)", "Segment", "Time"]
            my_df = my_df[cols]
            my_data_list.append(Voltammegram(data = my_df, name=sig_name))
            
            
        return VoltammegramCollection(
                signals=my_data_list,
                name=name
            )
        