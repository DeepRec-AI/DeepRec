#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .request import Request
from .request import Response
from . import tf_request_pb2 as tf_pb


class TFRequest(Request):
    """
    Request for tensorflow services whose input data is in format of protobuf,
    privide methods to generate the required protobuf object, and serialze it to string
    """
    DT_FLOAT = tf_pb.DT_FLOAT
    DT_DOUBLE = tf_pb.DT_DOUBLE
    DT_INT8 = tf_pb.DT_INT8
    DT_INT16 = tf_pb.DT_INT16
    DT_INT32 = tf_pb.DT_INT32
    DT_INT64 = tf_pb.DT_INT64
    DT_UINT8 = tf_pb.DT_UINT8
    DT_UINT16 = tf_pb.DT_UINT16
    DT_QINT8 = tf_pb.DT_QINT8
    DT_QUINT8 = tf_pb.DT_QUINT8
    DT_QINT16 = tf_pb.DT_QINT16
    DT_QUINT16 = tf_pb.DT_QUINT16
    DT_QINT32 = tf_pb.DT_QINT32
    DT_STRING = tf_pb.DT_STRING
    DT_BOOL = tf_pb.DT_BOOL

    def __init__(self, signature_name=None):
        self.request_data = tf_pb.PredictRequest()
        self.signature_name = signature_name

    def __str__(self):
        return self.request_data

    def set_signature_name(self, singature_name):
        """
        Set the signature name of the model
        :param singature_name: signature name of the model
        """
        self.signature_name = singature_name

    def add_feed(self, input_name, shape, content_type, content):
        """
        Add input data for the request, a tensorflow model may have many inputs with different
        data types, this methods set data for one of the input with the specified name 'input_name'
        :param input_name: name of the input to be set
        :param shape: shape of the input tensor in format of array, such as [1,784]
        :param content_type: type of the input tensor, can be one of the predefined data type, such as TFRequest.DT_FLOAT
        :param content: data content of the input tensor, which is expanded to one-dimension array, such as [1,2,3,4,5]
        """
        self.request_data.signature_name = self.signature_name
        self.request_data.inputs[input_name].dtype = content_type
        self.request_data.inputs[input_name].array_shape.dim.extend(shape)
        integer_types = [
            tf_pb.DT_INT8,
            tf_pb.DT_INT16,
            tf_pb.DT_INT32 ,
            tf_pb.DT_UINT8 ,
            tf_pb.DT_UINT16,
            tf_pb.DT_QINT8,
            tf_pb.DT_QINT16,
            tf_pb.DT_QINT32,
            tf_pb.DT_QUINT8,
            tf_pb.DT_QUINT16,
        ]
        if content_type == tf_pb.DT_FLOAT:
            self.request_data.inputs[input_name].float_val.extend(content)
        elif content_type == tf_pb.DT_DOUBLE:
            self.request_data.inputs[input_name].double_val.extend(content)
        elif content_type in integer_types:
            self.request_data.inputs[input_name].int_val.extend(content)
        elif content_type == tf_pb.DT_INT64:
            self.request_data.inputs[input_name].int64_val.extend(content)
        elif content_type == tf_pb.DT_BOOL:
            self.request_data.inputs[input_name].bool_val.extend(content)
        elif content_type == tf_pb.DT_STRING:
            self.request_data.inputs[input_name].string_val.extend(content)

    def add_fetch(self, output_name):
        """
        Add output node name for the request to get, if not specified, then all the known outputs are fetched,
        but for frozen models, the output name must be specified, or else the service would throw exception like:
        'Must specify at least one target to fetch or execute.'
        :param output_name: name of the output node to fetch
        """
        self.request_data.output_filter.append(output_name)

    def to_string(self):
        """
        Serialize the request to string for transmission
        :return: the request data in format of string
        """
        return self.request_data.SerializeToString()

    def parse_response(self, response_data):
        """
        Parse the given response data in string format to the related TFResponse object
        :param response_data: the service response data in string format
        :return: the TFResponse object related the request
        """
        return TFResponse(response_data)


class TFResponse(Response):
    """
    Deserialize the response data to a structured object for users to read
    """

    def __init__(self, response_data=None):
        self.response = tf_pb.PredictResponse()
        self.response.ParseFromString(response_data)

    def __str__(self):
        return str(self.response)

    def get_tensor_shape(self, output_name):
        """
        Get the shape of a specified output tensor
        :param output_name: name of the output tensor
        :return: the shape in format of list
        """
        return list(self.response.outputs[output_name].array_shape.dim)

    def get_values(self, output_name):
        """
        Get the value of a specified output tensor
        :param output_name: name of the output tensor
        :return: the content of the output tensor
        """
        output = self.response.outputs[output_name]
        if output.dtype == TFRequest.DT_FLOAT:
            return output.float_val
        elif output.dtype == TFRequest.DT_INT8 or output.dtype == TFRequest.DT_INT16 or \
                output.dtype == TFRequest.DT_INT32:
            return output.int_val
        elif output.dtype == TFRequest.DT_INT64:
            return output.int64_val
        elif output.dtype == TFRequest.DT_DOUBLE:
            return output.double_val
        elif output.dtype == TFRequest.DT_STRING:
            return output.string_val
        elif output.dtype == TFRequest.DT_BOOL:
            return output.bool_val

