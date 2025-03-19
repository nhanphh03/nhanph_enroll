import logging
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
from elasticsearch import Elasticsearch, exceptions

from config_run import Config
from modules import utils
from modules.driver.face_storage import FaceStorage

LOGGER = logging.getLogger("es")


class EsFaceStorageFactory:
    es_face_storage = None

    @staticmethod
    def get_instance():
        if EsFaceStorageFactory.es_face_storage is None:
            EsFaceStorageFactory.es_face_storage = EsFaceStorage(Config.es_hosts,
                                                                 Config.face_index,
                                                                 Config.similar_thresh_search,
                                                                 Config.max_output_search,
                                                                 Config.similar_thresh_compare,
                                                                 Config.similar_thresh_same)
        return EsFaceStorageFactory.es_face_storage


class EsFaceStorage(FaceStorage):
    def __init__(self, hosts: List[str], index: str = 'face',
                 similar_thresh_search: float = 0.6,
                 max_output_search: int = 10,
                 similar_thresh_compare: float = 0.8,
                 similar_thresh_same: float = 0.95):
        self._hosts = hosts
        self._index = index
        self._similar_thresh_search = similar_thresh_search
        self._max_output_search = max_output_search
        self._similar_thresh_compare = similar_thresh_compare
        self._similar_thresh_same = similar_thresh_same
        self._es = Elasticsearch(hosts=hosts,
                                 http_compress=True,
                                 send_get_body_as="POST")
        self._init_index(index)

    def _init_index(self, index):
        mapping = {
            "settings": {
                "number_of_shards": 3,
                "index": {
                    "knn": True,
                    "knn.space_type": "cosinesimil"
                }
            },
            "mappings": {
                "properties": {
                    "people_id": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "file_id": {
                        "type": "text"
                    },
                    "face": {
                        "type": "knn_vector",
                        "dimension": 512
                    },
                    "timestamp": {
                        "type": "date",
                        "format": "dd/MM/yyyy HH:mm:ss"
                    },
                    "source": {
                        "type": "text"
                    }
                }
            }
        }
        try:
            if not self._es.indices.exists(index=index, request_timeout=1):
                LOGGER.info("Create index ES with mapping")
                self._es.indices.create(index=index, body=mapping, request_timeout=1)
        except exceptions.ConnectionError as e:
            LOGGER.error(f"Can't connect to Open Distro: {self._hosts}")
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.info("Index has created.")

    def insert(self, file_id: str, people_id: str, face: np.ndarray, source: str, meta_data: Optional[Dict]) -> Dict:
        res = self._check_exists(face, source)
        if res is not None:
            if res.get('people_id') == people_id:
                return {
                    'status': 'DUPLICATE_REGISTER',
                    'message': f'Face has registered previously.',
                    'similarity_score': res.get('score'),
                    'old_people_id': res.get('people_id'),
                    'meta_data': res.get('meta_data')
                }
            else:
                return {
                    'status': 'ERROR',
                    'message': f'Face has registered with other id: "{res.get("people_id")}"',
                    'similarity_score': res.get('score'),
                    'old_people_id': res.get('people_id'),
                    'meta_data': res.get('meta_data')
                }

        data = {
            'file_id': file_id,
            'people_id': people_id,
            'face': face.tolist(),
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'meta_data': meta_data,
            'source': source
        }

        LOGGER.info(f"Insert new person '{people_id}' from source: '{source}'")
        res = self._es.index(index=self._index, body=data)
        return {
            'status': 'SUCCESS' if res['result'] == 'created' else 'ERROR',
            'message': 'Face has inserted to database.' if res['result'] == 'created' else 'Can not insert face to database.',
            'people_id': people_id,
            'source': source,
            'meta_data': meta_data
        }

    def _check_exists(self, face: np.ndarray, source: str) -> Optional[Dict]:
        res = self._search_vector(face, source, self._similar_thresh_same)
        if res is not None and len(res) > 0:
            return res[0]
        return None

    def _search_file_id(self, file_id: str) -> List[Dict]:
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "match_phrase": {
                                "file_id": file_id
                            }
                        }
                    ]
                }
            },
            "_source": ['people_id', 'source', 'meta_data']
        }
        return self._search_and_norm_res(query)

    def _search_id(self, people_id: str, source: str) -> List[Dict]:
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "match_phrase": {
                                "people_id": people_id
                            }
                        },
                        {
                            "match_phrase": {
                                "source": source
                            }
                        }
                    ]
                }
            },
            "_source": ['people_id', 'source', 'meta_data']
        }
        return self._search_and_norm_res(query)

    def _search_vector(self, face: np.ndarray, source: str, sim_threshold: Optional[float] = None) -> List[Dict]:
        if sim_threshold is None:
            sim_threshold = self._similar_thresh_compare
        query = {
            "min_score": self._similar_thresh_search,
            "_source": ['people_id', 'face', 'source', 'meta_data'],
            "size": self._max_output_search,
            "query": {
                "knn": {
                    "face": {
                        "k": self._max_output_search,
                        "vector": face.tolist()
                    }
                }
            },
            "post_filter": {
                "match": {
                    "source": source
                }
            }
        }
        res = self._es.search(index=self._index, body=query)
        LOGGER.info(f"Query: Got {res['hits']['total']['value']} Hits in {res['took']}ms")
        LOGGER.debug(f"Query: {res}")
        emb1 = np.array(face)
        results = []
        for f in res['hits']['hits']:
            emb2 = np.array(f['_source']['face'])
            score = utils.compute_sim_optimize(emb1, emb2, new_range=True)
            if score > sim_threshold:
                results.append({
                    '_id': f['_id'],
                    'people_id': f['_source'].get('people_id'),
                    'meta_data': f['_source'].get('meta_data'),
                    'source': f['_source'].get('source'),
                    'score': score
                })
                if score > 0.9:
                    break
            else:
                break
        return results

    def _search_meta_data(self, meta_data: Dict, source: str) -> List[Dict]:
        condition = []
        for k, v in meta_data:
            condition.append(
                {
                    "term": {
                        f"meta_data.{k}": v
                    }
                }
            )
        query = {
            "query": {
                "bool": {
                    "should": condition
                }
            }
        }
        return self._search_and_norm_res(query)

    def _search_and_norm_res(self, query: Dict) -> List:
        res = self._es.search(index=self._index, body=query)
        results = []
        for f in res['hits']['hits']:
            results.append({
                '_id': f['_id'],
                'people_id': f['_source'].get('people_id'),
                'meta_data': f['_source'].get('meta_data'),
                'source': f['_source'].get('source'),
                'score': None
            })
        return results

    def update_metadata(self, people_id: str, source: str, meta_data: Dict, file_id: str = None) -> int:
        res = self.search(people_id=people_id, face=None, source=source, meta_data=None, file_id=file_id)
        if res is not None and len(res) > 0:
            for r in res:
                data = {
                    'meta_data': meta_data,
                    'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                }

                res_updated = self._es.update(self._index, id=r['_id'], body={
                    'doc': data
                })

            return len(res)
        return 0

    def delete(self, people_id: str, source: str, file_id: str = None) -> int:
        filters = [
            {
                "match_phrase": {
                    "people_id": people_id
                }
            },
            {
                "match_phrase": {
                    "source": source
                }
            }
        ]
        if file_id is not None:
            filters.append({
                'match_phrase': {
                    "file_id": file_id
                }
            })
        res = self._es.delete_by_query(self._index, body={
            "query": {
                "bool": {
                    "filter": filters
                }
            }
        })
        return res['deleted']
