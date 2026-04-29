import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

class TNModel(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def format():
        pass


class tn4mlModel(TNModel):
    
