import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

class TNModel(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def format():
        pass

# Will be used to better structure the code and to make it more modular. This will be the base class for all the models that we will use in the future.