from typing import Dict, Optional, List

import numpy as np


class FaceStorage:

    def insert(self, file_id: str, people_id: str, face: np.ndarray, source: str, meta_data: Optional[Dict]) -> Dict:
        raise NotImplementedError("This method must be implemented in child class")

    def _check_exists(self, face: np.ndarray, source: str) -> Optional[Dict]:
        raise NotImplementedError("This method must be implemented in child class")

    def search(self, people_id: Optional[str],
               face: Optional[np.ndarray],
               source: str,
               meta_data: Optional[Dict],
               file_id: str = None) -> List[Dict]:
        if file_id is not None:
            return self._search_file_id(file_id)

        if people_id is not None:
            return self._search_id(people_id, source)

        if face is not None:
            return self._search_vector(face, source)

        if meta_data is not None:
            return self._search_meta_data(meta_data, source)

        raise ValueError("There isn't any information to search")

    def _search_file_id(self, file_id: str) -> List[Dict]:
        raise NotImplementedError("This method must be implemented in child class")

    def _search_id(self, people_id: str, source: str) -> List[Dict]:
        raise NotImplementedError("This method must be implemented in child class")

    def _search_vector(self, face: np.ndarray, source: str, sim_threshold: Optional[float] = None) -> List[Dict]:
        raise NotImplementedError("This method must be implemented in child class")

    def _search_meta_data(self, meta_data: Dict, source: str) -> List[Dict]:
        raise NotImplementedError("This method must be implemented in child class")

    def update_metadata(self, people_id: str, source: str, meta_data: Dict, file_id: str = None) -> bool:
        raise NotImplementedError("This method must be implemented in child class")

    def delete(self, people_id: str, source: str, file_id: str) -> bool:
        raise NotImplementedError("This method must be implemented in child class")
