import os
from typing import Mapping
from .signal import Signal

class SignalCollection:
    '''
    A SignalCollection contains multiple signals and is designed to 
    easily compare and visualize them. Dimensions of signals in a signal
    collection must be the same. Signals in a collection is stored by a
    dictionary, you can find every signal with its name like SignalCollection[signal_name]
    '''

    @staticmethod
    def merge(signal_collections: list['SignalCollection'], name="Merged_signal_collection") -> 'SignalCollection':
        signals = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals.values():
                signals.append(signal)
        return SignalCollection(signals, name=name)

    def __init__(self, signals: list[Signal] = [], name="Default_signal_collection") -> None:
        self.signals: Mapping[str, Signal] = {}
        for signal in signals:
            signal_name = signal.get_name()
            if not signal_name in self.signals.keys():
                self.signals[signal_name] = signal
            else:
                raise ValueError(f"Signal name {signal_name} already exists in the collection")
        self.name = name
        self.figsize = None
        self.colormap = "default"
        self.colormap_min = 0
        self.colormap_max = 1
    
    def keys(self):
        return self.signals.keys()
        
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
    
    def get_signal(self, signal_name: str) -> Signal:
        if signal_name not in self.signals.keys():
            raise ValueError(f"Signal name {signal_name} does not exist in the collection")
        return self.signals[signal_name]
    
    def __getitem__(self, signal_name: str):
        return self.get_signal(signal_name)
    
    def __delitem__(self, signal_name: str):
        del self.signals[signal_name]
    
    def add_signal(self, signal: Signal) -> None:
        signal_name = signal.get_name()
        if not signal_name in self.keys():
            self.signals[signal_name] = signal
        else:
            raise ValueError(f"Signal name {signal_name} already exists in the collection")
        
    def set_signal(self, signal_name, signal: Signal) -> None:
        self.signals[signal_name] = signal
    
    def __setitem__(self, signal_name: str, signal: Signal):
        self.set_signal(signal_name, signal)

    def rename_signal(self, old_signal_name, new_signal_name):
        if new_signal_name in self.keys():
            raise ValueError(f"Signal name {new_signal_name} already exists in the collection")
        self.signals[new_signal_name] = self.signals.pop(old_signal_name)
        self.signals[new_signal_name].set_name(new_signal_name)
    
    def export(self, root_directory, mode='write'):
        if not self.get_name():
            raise ValueError("Signal collection name is not set")
        true_directory = os.path.join(root_directory, self.get_name())
        directories_to_check = [root_directory, true_directory]
        for directory in directories_to_check:
            if not os.path.exists(directory):
                os.mkdir(directory)
        if os.path.exists(true_directory):
            if mode=="append":
                pass
            elif mode=="write":
                raise Exception("Directory already exists")
            elif mode=="replace":
                for file in os.listdir(directory):
                    os.remove(os.path.join(directory, file))
            else:
                raise ValueError(f"Invalid mode {mode}, should be either 'write', 'append' or 'replace'")
        flag = False
        for signal in self:
            if "/" in signal:
                print(f"Warning: signal name {signal} contains '/', which is not allowed in file names")
                flag = True
        if flag:
            raise ValueError("Signal names contain '/'")
        for signal in self.signals.values():
            signal.export(os.path.join(directory, signal.get_name() + ".csv"), mode=mode)
    
    def copy(self):
        signals: list[Signal] = []
        for signal in self.signals.values():
            signals.append(signal.copy())
        return type(self)(signals, name=self.get_name())
    
    def __iter__(self):
        return iter(self.signals)
    
    def __len__(self):
        return len(self.signals)
    
    def set_figsize(self, figsize):
        self.figsize = figsize
    
    def set_colormap(self, colormap_name, colormap_min = 0, colormap_max = 1):
        '''
        colormap_name: str or list. str for predefined colormap, list for custom color lists
        '''
        if isinstance(colormap_name, list):
            if len(colormap_name) != len(self):
                raise ValueError("The length of colormap_name should be the same as the length of the collection")
        self.colormap = colormap_name
        self.colormap_min = colormap_min
        self.colormap_max = colormap_max