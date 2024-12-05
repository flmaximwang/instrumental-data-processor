import re
import io
import pandas as pd

from ..concretes.absorb_spec import AbsorbSpec, AbsorbSpecCollection
from ..abstracts import ContinuousSignal1DCollection
from ..utils import path_utils, name_utils

class NanodropWorkbook(AbsorbSpecCollection):
    
    @staticmethod
    def from_exported_file(file_path, name=None):
        spectrums = []
        # 确保文件的末尾是 3 个换行符, 以便最后一个 spectrum 能被正确读取
        with open(file_path, "r") as file_input:
            lines = file_input.readlines()
        while lines.pop() == "\n":
            continue
        for i in range(3):
            lines.extend(["\n"])
        with open(file_path, "w") as file_output:
            file_output.writelines(lines)
            
        with open(file_path) as file_input:
            lines = file_input.readlines()
        # 将 lines 分割, 遇到 2 个连续换行符时进行分割
        new_line_indices = [i for i, line in enumerate(lines) if line == "\n"]
        separator_indices = [new_line_indices[i] for i in range(0, len(new_line_indices) - 1) if new_line_indices[i + 1] - new_line_indices[i] == 1 and new_line_indices[i] - new_line_indices[i - 1] != 1]
        separator_indices.insert(0, -2)
        for i in range(len(separator_indices) - 1):
            # print(lines[separator_indices[i] + 2:separator_indices[i + 1]])
            entryName = lines[separator_indices[i] + 2][:-1]
            time = lines[separator_indices[i] + 3][:-1]
            csv_text = "".join(lines[separator_indices[i] + 4:separator_indices[i + 1]])
            data = pd.read_csv(io.StringIO(csv_text), sep="\t", dtype=float)
            spectrums.append(AbsorbSpec(data, f"{entryName}_{time}", axis_name = "Wavelength", axis_unit="nm", value_name=data.columns[1]))
        name_counter = {}
        for i, spectrum in enumerate(spectrums):
            spectrum_name = spectrum.get_name()
            if spectrum_name not in name_counter:
                name_counter[spectrum_name] = 0
            else:
                name_counter[spectrum_name] += 1
                spectrum.set_name(spectrum_name + "_" + str(name_counter[spectrum_name]))
                
        return NanodropWorkbook(signals = spectrums, name=name)

    def remove_timestamp(self):
        original_sig_names = list(self.keys())
        new_sig_names = ["_".join(sig_name.split("_")[:-1]) for sig_name in original_sig_names]
        name_utils.rename_duplicated_names(new_sig_names)
        for new_name, old_name in zip(new_sig_names, original_sig_names):
            self.rename_signal(old_name, new_name)