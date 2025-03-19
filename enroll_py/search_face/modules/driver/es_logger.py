"""
Logger request to elastic_search
"""
import json
import logging
import re
import uuid
from urllib.parse import urlparse

from config_run import Config
from modules.driver.elasticsearch_driver import ElasticSearchDriver

LOGGER = logging.getLogger("api")


class EsLogger:

    def __init__(self, config: Config):
        """

        :param config: host of elastic search
        """
        self._es_driver = ElasticSearchDriver(hosts=config.es_hosts, index='logger')
        self._index = config.api_log_index

    def log_request(self, endpoint: str,
                    method: str,
                    timestamp: str,
                    duration: float,
                    ip_address: str,
                    app_id: str,
                    host_address: str,
                    input_body: str,
                    status_code: int,
                    status: str,
                    response: dict):
        request_id = uuid.uuid4()
        if response is not None and "status_code" in response:
            res_status_code = response['status_code']
        else:
            res_status_code = None

        image_source = None
        if input_body is not None:
            url = re.search(
                r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
                input_body)
            if url is not None:
                for x in url.regs:
                    url = input_body[x[0]:x[1]]
                    parser_result = urlparse(url)
                    image_source = parser_result.hostname
                    break

        data = {
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "timestamp": timestamp,
            "duration": duration,
            "ip_address": ip_address,
            "app_id": app_id,
            "host_address": host_address,
            "input_body": input_body,
            "img_src": image_source,
            "status_code": status_code,
            "status": status,
            "res_status_code": res_status_code,
            "response": json.dumps(response, ensure_ascii=False)
        }
        res = self._es_driver.insert(self._index, data)
        LOGGER.debug(f"Result logging to api: {res}")
        return request_id
