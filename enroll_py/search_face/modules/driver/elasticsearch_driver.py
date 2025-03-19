import json
import logging
from datetime import datetime
from typing import List
import numpy as np

from elasticsearch import Elasticsearch, helpers, exceptions

from config_run import Config
from modules import utils
from modules.utils import encode_array

LOGGER = logging.getLogger("es")


class ElasticSearchDriver:
    def __init__(self, hosts, index: str = None):
        self._es = Elasticsearch(hosts,
                                 http_compress=True,
                                 send_get_body_as="POST")
        self._hosts = hosts
        self._init_index(index)

    def _init_index(self, index: str):
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
                    "face": {
                        "type": "knn_vector",
                        "dimension": 512
                    },
                    "timestamp": {
                        "type": "date",
                        "format": "dd/MM/yyyy HH:mm:ss"
                    },
                    "created_at": {
                        "type": "text"
                    },
                    "source": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
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

    def query(self, index, body):
        res = self._es.search(body=body, index=index)
        return res['hits']['hits']

    def insert(self, index, data):
        try:
            return self._es.index(index=index, body=data)
        except Exception as e:
            LOGGER.exception(e)
            return None

    def insert_face(self, index: str, people_id: str, face_emb, created_at: str):
        face_emb = encode_array(face_emb)
        res = self._es.exists(index=index, id=people_id)
        data = {
            'people_id': people_id,
            'face': face_emb,
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'created_at': created_at
        }
        if res:
            return self._es.update(index=index, id=people_id, body={
                "doc": data
            })
        return self._es.index(index=index, id=people_id, body=data)

    def insert_face2(self, index: str, people_id: str, face_emb, created_at: str, source: str):
        body_check_exists = {
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
            "_source": ['people_id', 'source', 'create_at']
        }
        res = self._es.search(index=index, body=body_check_exists)
        data = {
            'people_id': people_id,
            'face': face_emb.tolist(),
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'created_at': created_at,
            'source': source
        }
        if res['hits']['total']['value'] > 0:
            LOGGER.info(f"Exist person '{people_id}' from source: '{source}': {res['hits']['hits']}")
            LOGGER.info(f"Update person '{people_id}' from source: '{source}'")
            doc_id = res['hits']['hits'][0]['_id']
            return self._es.update(index=index, id=doc_id, body={
                "doc": data
            })

        LOGGER.info(f"Insert new person '{people_id}' from source: '{source}'")
        return self._es.index(index=index, body=data)

    def insert_bulk(self, data):
        """
        Insert batch
        :param data: list object
        :return:
        """
        try:
            return helpers.bulk(self._es, self._create_generator(data), chunk_size=100, request_timeout=200,
                                max_retries=3)
        except Exception as e:
            LOGGER.exception(e)
            return None

    def insert_bulk_parallel(self, data):
        """
        Insert quickly with parallel bulk
        :param data: list object
        :return:
        """
        try:
            return helpers.parallel_bulk(self._es, self._create_generator(data), thread_count=4, chunk_size=500,
                                         max_chunk_bytes=100 * 1024 * 1024, queue_size=4)
        except Exception as e:
            LOGGER.exception(e)
            return None

    def search(self, index, body):
        res = self._es.search(index=index, body=body)
        LOGGER.info(f"Search({index}): Got {res['hits']['total']['value']} hits")
        return res['hits']['hits']

    def query_by_id(self, index, id):
        """
        Query record face in elastic search people id
        :param index:
        :param id:
        :return:
        """
        query = {
            "query": {
                "bool": {
                    "must": [{
                        "term": {
                            "people_id": id
                        }
                    }]
                }
            },
            "size": 10
        }
        res = self._es.search(index=index, body=query)
        LOGGER.info(f"query_by_id({id}): Got {res['hits']['total']['value']} Hits")
        return res['hits']['hits']

    def query_embedding(self, index, embedding, outputs=None,
                        min_score: float = 0.8,
                        max_date: str = None,
                        max_outputs: int = 10):
        """
        Query face has embedding same with input embedding
        :param max_date: filter max date of record
        :param index: index database in es
        :param embedding: list 512 float, vector embedding of face
        :param outputs: list field to return
        :param min_score:
        :param max_outputs: maximum record to return
        :return: list of records
        """
        if outputs is None:
            outputs = ["people_id"]
        query = {
            "min_score": min_score,
            "query": {
                "function_score": {
                    "boost_mode": "replace",
                    "script_score": {
                        "script": {
                            "source": "binary_vector_score",
                            "lang": "knn",
                            "params": {
                                "cosine": True,
                                "field": "face",
                                "vector": embedding
                            }
                        }
                    }
                }
            },
            "_source": outputs,
            "size": max_outputs
        }

        if max_date:
            sub_query = {
                "bool": {}
            }
            if max_date:
                sub_query["bool"]["filter"] = [
                    {
                        "match_all": {}
                    },
                    {
                        "range": {
                            "create_on": {
                                "format": "strict_date_optional_time",
                                "lt": max_date  # "2020-08-08T16:59:59.999Z"
                            }
                        }
                    }
                ]
            query['query']['function_score']['query'] = sub_query
        # print("Query: {}".format(query))
        res = self._es.search(index=index, body=query)
        LOGGER.info(f"Query: Got {res['hits']['total']['value']} Hits")

        return res['hits']['hits']

    def query_embedding2(self, index, embedding, source: str, outputs=None,
                         min_score: float = 0.8,
                         max_date: str = None,
                         max_outputs: int = 10):
        """
        Query face has embedding same with input embedding
        :param max_date: filter max date of record
        :param index: index database in es
        :param embedding: list 512 float, vector embedding of face
        :param source: source of result to query
        :param outputs: list field to return
        :param min_score:
        :param max_outputs: maximum record to return
        :return: list of records
        """
        if outputs is None:
            outputs = ["people_id"]
        outputs.append('face')
        query = {
            "min_score": Config.similar_thresh_search,
            "_source": outputs,
            "size": max_outputs,
            "query": {
                "knn": {
                    "face": {
                        "k": max_outputs,
                        "vector": embedding
                    }
                }
            },
            "post_filter": {
                "match": {
                    "source": source
                }
            }
        }

        # LOGGER.debug("Query: {}".format(query))
        res = self._es.search(index=index, body=query)
        LOGGER.info(f"Query: Got {res['hits']['total']['value']} Hits in {res['took']}ms")

        emb1 = np.array(embedding)
        results = []
        for f in res['hits']['hits']:
            emb2 = np.array(f['_source']['face'])
            score = utils.compute_sim(emb1, emb2, new_range=True)
            if score > min_score:
                tmp = dict(f)
                del tmp['_source']['face']
                tmp['_score'] = score
                results.append(tmp)
                if score > 0.9:
                    break
            else:
                break
        return results

    def query_embedding_multi(self, index: str, ids: List[str], embeddings: List[List[float]], sources: List[str],
                              outputs=None,
                              min_score: float = 0.8,
                              max_date: str = None,
                              max_outputs: int = 10):
        """
        Query face has embedding same with input embedding
        :param max_date: filter max date of record
        :param index: index database in es
        :param ids: list id of requests
        :param embeddings: list 512 float, vector embedding of face
        :param sources: List source of result to query
        :param outputs: list field to return
        :param min_score:
        :param max_outputs: maximum record to return
        :return: list of records
        """
        if outputs is None:
            outputs = ["people_id"]
        outputs.append('face')
        queries = []
        for embedding, source in zip(embeddings, sources):
            queries.append({'index': index})
            query = {
                "min_score": Config.similar_thresh_search,
                "_source": outputs,
                "size": max_outputs,
                "query": {
                    "knn": {
                        "face": {
                            "k": max_outputs,
                            "vector": embedding
                        }
                    }
                },
                "post_filter": {
                    "match": {
                        "source": source
                    }
                }
            }
            queries.append(query)
        queries = [json.dumps(q) for q in queries]
        query_str = '\n'.join(queries)
        # LOGGER.debug("Query: {}".format(query_str))
        res = self._es.msearch(index=index, body=query_str)
        # LOGGER.info(f"Result Query: {res}")
        # LOGGER.info(f"Query: Got {res['hits']['total']['value']} Hits")

        embeddings = np.array(embeddings)
        results = {}
        for i, (r, img_id) in enumerate(zip(res['responses'], ids)):
            result_face = []
            for f in r['hits']['hits']:
                emb2 = np.array(f['_source']['face'])
                score = utils.compute_sim(embeddings[i], emb2, new_range=True)
                if score > min_score:
                    result_face.append({
                        'people_id': f['_source']['people_id'],
                        'created_at': f['_source']['created_at'],
                        'source': f['_source']['source'],
                        'score': score
                    })
                    if score > 0.9:
                        break
                else:
                    break
            results[img_id] = result_face
        return results

    @staticmethod
    def _create_generator(data):
        """
        :param data: list of object
        :return:
        """
        for d in data:
            yield d

    def update(self, data):
        """
        Update data in elastic search
        :param data: list object
        :return:
        """
        try:
            success, info = helpers.bulk(self._es, self._create_generator(data), chunk_size=100, request_timeout=200)
            if not success:
                LOGGER.error("A document failed:", info)
            return success, info
        except Exception as e:
            LOGGER.exception(e)
            print(e)
            return None

    def update_parallel(self, data):
        """
        Update quickly data in elastic search
        :param data: list object
        :return:
        """
        try:
            return helpers.parallel_bulk(self._es, self._create_generator(data), thread_count=4, chunk_size=500,
                                         max_chunk_bytes=100 * 1024 * 1024, queue_size=4)
        except Exception as e:
            LOGGER.exception(e)
            return None

    def delete_people(self, index: str, people_id: str, source: str):
        data = {
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
            }
        }
        LOGGER.info(f"Delete person '{people_id}' from source '{source}'")
        return self._es.delete_by_query(index=index, body=data)


INSTANCE: ElasticSearchDriver = None


def initialize_driver(hosts: List[str], index: str = None):
    global INSTANCE
    if INSTANCE is None:
        if index is None:
            index = 'face'
        INSTANCE = ElasticSearchDriver(hosts, index)


def get_instance():
    global INSTANCE
    if INSTANCE is None:
        raise Exception('Must be initialize ES driver before.')
    return INSTANCE
