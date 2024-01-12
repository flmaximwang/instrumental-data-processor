import re
import io
import pandas as pd

from ..concretes.absorb_spec import AbsorbSpec
from ..abstracts import Signal1DCollection
from ..utils import path_utils

class NanodropWorkbook(Signal1DCollection):
    
    @staticmethod
    def from_exported_file(file_path, name=None):
        spectrums = []
        # 确保文件的末尾存是 3 个换行符, 以便最后一个 spectrum 能被正确读取
        with open(file_path, "r") as file_input:
            # 首先读取文件的最后 3 个字符, 如果都是换行符, 则略过; 如果只有 2 个换行符, 则写入 1 个换行符
            lines = file_input.readlines()
            last_char = lines[-1]
        if last_char != "\n":
            with open(file_path, "a") as file_input:
                file_input.write("\n")
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
                        data = pd.read_csv(io.StringIO(csv_text))
                        spectrums.append(AbsorbSpec(
                            data=data,
                            name=entryName,
                            axis_name="Wavelength",
                            axis_unit="nm",
                            value_name=data.columns[1],
                            value_unit=None,
                        ))
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
            
        # 分析 spectrums 中的每一个光谱, 避免名称重复; 重复者后面加上序号
        name_counter = {}
        for i, spectrum in enumerate(spectrums):
            spectrum_name = spectrum.get_name()
            if spectrum_name not in name_counter:
                name_counter[spectrum_name] = 0
            else:
                name_counter[spectrum_name] += 1
                spectrum.set_name(spectrum_name + "_" + str(name_counter[spectrum_name]))
                
        return Signal1DCollection(signals = spectrums, name=name)