from ..concretes import MultiWellPlate
import io
import pandas as pd

class MultiWellPlateFC(MultiWellPlate):
    
    @staticmethod
    def from_exported_file(txt_path, name="Default Multi-Scan Fluorescence Collection"):
        with open(txt_path, 'r') as f:
            flag = False
            data_txt = ""
            for line in f:
                if not flag and "	1	2	3	4	5	6	7	8	9	10	11	12" in line:
                    flag = True
                if flag:
                    data_txt += line
        my_df = pd.read_csv(io.StringIO(data_txt), sep="\t", index_col=0)
        
        return MultiWellPlateFC(data=my_df, name=name)