import logging
import numpy as np
from typing import Tuple, Dict, Any

import requests
from grpc import StatusCode
from grpc._channel import _InactiveRpcError
from requests import ConnectTimeout

from modules.inference.requests_tf import TensorServingClient
from protos.tensorflow_serving.apis.predict_pb2 import PredictResponse
from tensors_tf import tensor_proto_to_ndarray


class InputFormatException(Exception):
    pass


class ResponseFormatException(Exception):
    pass


class ServerModelException(Exception):
    pass


class BaseModelTF:
    LOGGER = logging.getLogger("model")

    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None, options: list = None,
                 timeout: int = 5, debug: bool = False):
        """
        Create client inference model server
        :param service_host: ip address of server model
        :param service_port: port of server model
        :param model_name: name of model
        :param version: version of model
        :param options: other option
        :param timeout: timeout when inference
        :param debug: debug
        """
        self._service_host = service_host
        self._service_port = service_port
        self._model_name = model_name
        self._version = version
        self._timeout = timeout
        self._debug = debug

        if version:
            self.http_url = f"http://{service_host}:{service_port + 1}/v1/models/{model_name}/versions/{version}"
        else:
            self.http_url = f"http://{service_host}:{service_port + 1}/v1/models/{model_name}"

        self._input_signature_key = []
        self._output_signature_key = []

        self._get_meta_data()
        self.tensor_serving_client: TensorServingClient = TensorServingClient(host=service_host,
                                                                              port=service_port,
                                                                              options=options)

    def _get_meta_data(self):
        url_metadata = f"{self.http_url}/metadata"
        try:
            res = requests.get(url_metadata, timeout=10)
            metadata = res.json()
            self._model_name = metadata['model_spec']['name']
            self._version = int(metadata['model_spec']['version'])
            inputs = metadata['metadata']['signature_def']['signature_def']['serving_default']['inputs']
            outputs = metadata['metadata']['signature_def']['signature_def']['serving_default']['outputs']
            for k, v in inputs.items():
                self._input_signature_key.append(
                    (k, v['dtype'], [int(dim['size']) for dim in v['tensor_shape']['dim']]))
            for k, v in outputs.items():
                self._output_signature_key.append(
                    (k, v['dtype'], [int(dim['size']) for dim in v['tensor_shape']['dim']]))

            self._input_signature_key = sorted(self._input_signature_key, key=lambda x: x[0])
            self._output_signature_key = sorted(self._output_signature_key, key=lambda x: x[0])

            self.LOGGER.info(
                f"Get MetaData from {self._model_name}:v{self._version}: input: {self._input_signature_key}   output: {self._output_signature_key}")
        except ConnectTimeout as e:
            self.LOGGER.error("Can not connect to {}.".format(url_metadata))
            raise ServerModelException("{}: {}: {}".format(self.__class__.__name__,"Can't connect to model", url_metadata))
        except KeyError:
            self.LOGGER.error("Can't find model in server. {}".format(metadata))
            raise ServerModelException("{}: {}: {}".format(self.__class__.__name__, ": Can't find model in server.", metadata))

    def predict(self, x, params: dict = None):
        input_dict, params_process = self._pre_process(x)
        if params is not None:
            params_process.update(params)
        try:
            predict_response: PredictResponse = self._get(input_dict)
        except _InactiveRpcError as e:
            # self.LOGGER.exception(e)
            if e.code() == StatusCode.DEADLINE_EXCEEDED:
                raise ServerModelException("{}: {}".format(self.__class__.__name__, "Timeout when inference model."))
            elif e.code() == StatusCode.INVALID_ARGUMENT:
                raise InputFormatException("{}: {}".format(self.__class__.__name__, e.details()))
            else:
                raise ServerModelException("{}: {}.\n{}".format(self.__class__.__name__, e, "Error when inference model."))

        except Exception as e:
            self.LOGGER.exception(e)
            raise ServerModelException("{}: {}".format(self.__class__.__name__, ": Error when inference model."))
        post_processed = self._post_process(predict_response, params_process)
        return self._normalize_output(post_processed, params_process)

    def _pre_process(self, x) -> Tuple[Dict[Any, np.ndarray], dict]:
        """
        Preprocess input and return dict of input
        :param x: input
        :return: dict of input for inference, dict of parameter process -> use to post process
        """
        raise NotImplementedError("_pre_process must be implement on child class")

    def _get(self, input_dict: dict) -> dict:
        res = self.tensor_serving_client.predict_request(model_name=self._model_name, input_dict=input_dict,
                                                         timeout=self._timeout,
                                                         model_version=self._version)
        predict_response = {}
        for k, t, shape in self._output_signature_key:
            predict_response[k] = tensor_proto_to_ndarray(res.outputs[k])
        return predict_response

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        """
        :param predict_response: dict predict response
        :param params_process: dict of param from preprocess
        :return:
        """
        raise NotImplementedError("_post_process must be implement on child class")

    def _normalize_output(self, post_processed: dict, params: dict):
        """
        Normalize output
        :param post_processed: result after post process
        :param params: parameters to format
        :return:
        """
        raise NotImplementedError("_normalize_output must be implement on child class.")
