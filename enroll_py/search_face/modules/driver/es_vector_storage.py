from datetime import datetime
import logging
from typing import List, Optional

import numpy as np
from elasticsearch import Elasticsearch, exceptions

from config_run import Config
from modules.driver.vector_storage import VectorStorage

LOGGER = logging.getLogger("es")


class EsVectorStorageFactory:
    _es_vector_storage = None

    @staticmethod
    def get_instance():
        if EsVectorStorageFactory._es_vector_storage is None:
            EsVectorStorageFactory._es_vector_storage = EsVectorStorage(Config.es_hosts, Config.vector_index)
        return EsVectorStorageFactory._es_vector_storage


class EsVectorStorage(VectorStorage):
    def __init__(self, hosts: List[str], index: str = 'vector'):
        self._caches = {}
        self._keys = []
        self._max_cache = 100
        self._hosts = hosts
        self._index = index
        self._es = Elasticsearch(hosts=hosts,
                                 http_compress=True,
                                 send_get_body_as="POST")
        self._init_index(index)

    def _init_index(self, index: str):
        query = {
            "settings": {
                "number_of_shards": 3
            },
            "mappings": {
                "properties": {
                    "file_id": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 512
                    },
                    "timestamp": {
                        "type": "date",
                        "format": "dd/MM/yyyy HH:mm:ss"
                    },
                    "source": {
                        "type": "text",
                    }
                }
            }
        }

        try:
            if not self._es.indices.exists(index=index, request_timeout=1):
                LOGGER.info("Create index ES with mapping")
                self._es.indices.create(index=index, body=query, request_timeout=1)
        except exceptions.ConnectionError as e:
            LOGGER.error(f"Can't connect to Open Distro: {self._hosts}")
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.info("Index has created.")

    def add_vector(self, id: str, vector: np.ndarray) -> bool:
        self._update_cache(id, vector)
        return self._insert(id, vector)

    def _update_cache(self, id: str, vector: np.ndarray):
        if len(self._keys) >= self._max_cache:
            last_key = self._keys.pop(0)
            self._caches.pop(last_key)

        self._keys.append(id)
        self._caches[id] = vector

    def _insert(self, id: str, vector: np.ndarray) -> bool:
        body_check_exists = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "match_phrase": {
                                "file_id": id
                            }
                        },
                        {
                            "match_phrase": {
                                "source": 'upload'
                            }
                        }
                    ]
                }
            },
            "_source": ['people_id', 'source', 'timestamp']
        }
        res_check_exists = self._es.search(index=self._index, body=body_check_exists)
        if res_check_exists['hits']['total']['value'] > 0:
            return False

        data = {
            'file_id': id,
            'vector': vector.tolist(),
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'source': 'upload'
        }
        res = self._es.index(self._index, body=data)
        return res['_shards']['successful'] > 0

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        vector = self._caches.get(id, None)
        if vector is not None:
            return vector

        body_check_exists = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "match_phrase": {
                                "file_id": id
                            }
                        },
                        {
                            "match_phrase": {
                                "source": 'upload'
                            }
                        }
                    ]
                }
            },
            "_source": ['file_id', 'vector']
        }
        res_check_exists = self._es.search(index=self._index, body=body_check_exists)
        if res_check_exists['hits']['total']['value'] == 0:
            return None

        vector = res_check_exists['hits']['hits'][0]['_source']['vector']
        vector = np.array(vector)
        self._update_cache(id, vector)
        return vector

    def get_last_id(self, n: int = 5) -> List:
        return self._keys[-5:]

    def remove_vector(self, id: str) -> bool:
        pass
