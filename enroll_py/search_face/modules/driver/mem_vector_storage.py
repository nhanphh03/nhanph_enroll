import numpy as np

from modules.driver.vector_storage import VectorStorage


class MemVectorStorage(VectorStorage):
    def __init__(self):
        self._db = {}

    def add_vector(self, id: str, vector: np.ndarray) -> bool:
        if id in self._db:
            return False
        else:
            self._db[id] = vector
            print(f"Size db increase to: {len(self._db)}")
            return True

    def get_vector(self, id: str) -> np.ndarray:
        if id in self._db:
            return self._db[id]
        else:
            return None

    def remove_vector(self, id: str) -> bool:
        if id in self._db:
            del self._db[id]
            return True
        else:
            return False
