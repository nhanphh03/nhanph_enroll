from typing import List, Dict
import numpy as np


class Searcher(object):
    def __init__(self):
        self.total_people = 0
        self.search_index = None
        self.list_face_idx_with_face_id = []
        self.face_id_info = []
        self.face_id_feature = []

    def _load_index(self):
        pass

    def insert(self, features: List[np.ndarray], face_ids: List[str], infos: List[Dict]) -> int:
        """
        Insert new features vector with info and id to index of searcher
        Args:
            features: List array float feature vector of face
            face_ids: List string face id
            infos: List dict other info of face
        Return:
            Number of record is inserted
        """
        raise NotImplementedError("insert must be implemented on child class")

    def search(self, feature: np.ndarray, k: int = 5, norm_score: bool = True):
        """
        Get top k record similar with input feature vector

        """
        raise NotImplementedError("search must be implemented on child class")

    def delete(self, face_ids: List):
        raise NotImplementedError("delete must be implemented on child class")
