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

    def __init__(self, signals: list[Signal] = [], name="Default_signal_collection") -> None:
        self.signals: Mapping[str, Signal] = {}
        for signal in signals:
            signal_name = signal.get_name()
            if not signal_name in self.signals.keys():
                self.signals[signal_name] = signal
            else:
                raise ValueError(f"Signal name {signal_name} already exists in the collection")
        self.name = name
        
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
    
    def add_signal(self, signal: Signal) -> None:
        signal_name = signal.get_name()
        if not signal_name in self.signals.keys():
            self.signals[signal_name] = signal
        else:
            raise ValueError(f"Signal name {signal_name} already exists in the collection")
        
    def set_signal(self, signal_name, signal: Signal) -> None:
        self.signals[signal_name] = signal
    
    def __setitem__(self, signal_name: str, signal: Signal):
        self.set_signal(signal_name, signal)

    def rename_signal(self, old_signal_name, new_signal_name):
        if new_signal_name in self.signals.keys():
            raise ValueError(f"Signal name {new_signal_name} already exists in the collection")
        self.signals[new_signal_name] = self.signals.pop(old_signal_name)
        self.signals[new_signal_name].set_name(new_signal_name)
    
    def export(self, directory, mode='write'):
        if not os.path.exists(directory):
            os.mkdir(directory)
        true_directory = directory = os.path.join(directory, self.get_name())
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
        for signal in self.signals.values():
            signal.export(os.path.join(directory, signal.get_name() + ".csv"), mode=mode)