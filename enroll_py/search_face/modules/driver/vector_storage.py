from typing import List

import numpy as np


class VectorStorage:

    def add_vector(self, id: str, vector: np.ndarray) -> bool:
        raise NotImplementedError("This method must be implemented in child class")

    def get_vector(self, id: str) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in child class")

    def get_last_id(self, n: int = 5) -> List:
        raise NotImplementedError("This method must be implemented in child class")

    def remove_vector(self, id: str) -> bool:
        raise NotImplementedError("This method must be implemented in child class")
