import re
import io
import pandas as pd

from ..concretes.absorb_spec import AbsorbSpec
from ..abstracts import Signal1DCollection
from ..utils import path_utils

class AbsorbSpecColl(Signal1DCollection):
    
    @staticmethod
    def from_exported_file(file_path, name=None):
        spectrums = []
        with open(file_path) as file_input:
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
                        spectrums.append(AbsorbSpec(pd.read_csv(io.StringIO(csv_text)), entryName, time))
                        csv_text = ""
                    else:
                        csv_text += line.replace("\t", ",")
                else:
                    raise ValueError("Unknown flag: " + flag)
            
            # append the last one
            if flag == "content":
                spectrums.append(AbsorbSpec(pd.read_csv(io.StringIO(csv_text)), entryName, time))
        
        if not name:
            name = path_utils.get_name_from_path(file_path)
                
        return AbsorbSpecColl(signals = spectrums, name=name)