import logging
import grpc
import numpy as np

from protos.tensorflow.core.framework import tensor_pb2
from protos.tensorflow.core.framework import tensor_shape_pb2
from protos.tensorflow.core.framework import types_pb2
from protos.tensorflow_serving.apis import predict_pb2
from protos.tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import requests


LOGGER = logging.getLogger("inference")


class ServiceException(Exception):
    pass


class TypeNotSupportException(Exception):
    pass


class ServiceTFConnection:
    mapping_input_type = {
        "DT_INT64": types_pb2.DT_INT64,
        "DT_UINT8": types_pb2.DT_UINT8,
        "DT_FLOAT": types_pb2.DT_FLOAT,
        "DT_STRING": types_pb2.DT_STRING
    }

    mapping_output_type = {
        "DT_UINT8": types_pb2.DT_UINT8,
        "DT_FLOAT": types_pb2.DT_FLOAT,
        "DT_STRING": types_pb2.DT_STRING
    }

    def __init__(self, service_host,
                 service_port,
                 model_name,
                 version=None,
                 options=[],
                 timeout=5, debug=False):
        self._service_host = service_host
        self._service_port = service_port
        self._model_name = model_name
        self._version = version
        self._options = options
        self._timeout = timeout
        self._debug = debug
        if version:
            self.http_url = f"http://{service_host}:{service_port + 1}/v1/models/{model_name}/versions/{version}"
        else:
            self.http_url = f"http://{service_host}:{service_port + 1}/v1/models/{model_name}"
        self._input_signature_key = []
        self._output_signature_key = []
        self._set_up_metadata()
        self._create_connection()

    def _create_connection(self):
        target = f"{self._service_host}:{self._service_port}"
        channel = grpc.insecure_channel(target, options=self._options)
        self._stub = PredictionServiceStub(channel)

    def _set_up_metadata(self):
        url_metadata = f"{self.http_url}/metadata"
        res = requests.get(url_metadata, timeout=10)
        metadata = res.json()
        try:
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

            print(f"Get MetaData from {self._model_name}:v{self._version}: input: {self._input_signature_key}   output: {self._output_signature_key}")
        except KeyError:
            print("Can not connect to {}: {}".format(url_metadata, metadata))
            exit(0)

    @staticmethod
    def make_tensor(input_tensor, dtype=types_pb2.DT_FLOAT):
        try:
            dims = input_tensor.ndim
        except:
            input_tensor = np.array([input_tensor])
            dims = input_tensor.ndim

        if dims == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)  # we do batch inference, so input is a 4-D tensor
        tensor_shape = input_tensor.shape
        dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape]
        tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)

        if dtype == types_pb2.DT_FLOAT:
            tensor = tensor_pb2.TensorProto(
                dtype=dtype,
                tensor_shape=tensor_shape,
                float_val=input_tensor.reshape(-1)
            )
        elif dtype in [types_pb2.DT_UINT8, types_pb2.DT_INT64]:
            tensor = tensor_pb2.TensorProto(
                dtype=dtype,
                tensor_shape=tensor_shape,
                int_val=input_tensor.reshape(-1)
            )
        else:
            raise TypeNotSupportException(f"Input type {dtype} is not support.")
        return tensor

    def _sent_request(self, inputs: list):
        """
        List input with other is the same signature
        :param inputs:
        :return:
        """

        # Prepare request object
        _request = predict_pb2.PredictRequest()
        _request.model_spec.name = self._model_name
        if self._version:
            _request.model_spec.version.value = self._version
        _request.model_spec.signature_name = "serving_default"  # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        input_signature = self._input_signature_key
        for signature, value in zip(input_signature, inputs):
            tensor = self.make_tensor(value, self.mapping_input_type[signature[1]])
            _request.inputs[signature[0]].CopyFrom(tensor)

        # Do inference
        res = self._stub.Predict.future(_request, self._timeout)  # 5s timeout
        result = res.result().outputs
        # Cast type of response
        responses = {}
        for k, t, shape in self._output_signature_key:
            v = result[k]
            shape = [int(x.size) for x in v.tensor_shape.dim]
            if t == 'DT_FLOAT':
                responses[k] = np.reshape(v.float_val, shape)
            elif t in ['DT_UINT8', 'DT_INT32']:
                responses[k] = np.reshape(v.int_val, shape)
            elif t == 'DT_STRING':
                responses[k] = v.string_val
            else:
                raise TypeError("Not define type: ", t)
        return responses

    def predict(self, inputs: list):
        """

        :param inputs: list of input with order signatures
        :return:
        """
        try:
            response = self._sent_request(inputs)
        except Exception as e:
            LOGGER.error(f"Can not inference to service model at: {self._service_host}:{self._service_port} : {e}")
            raise ServiceException("Service is not available.")
        return response