import math
import numpy as np
import onnx
from onnx import numpy_helper

from utils import _dict
class BaseNode:
    def __init__(self, node):
        self.node = node
        self.input = []
        self.output = []
        self.input_blocks = []
        self.output_blocks = []
        self.attributes = _dict()
        self._shape = None
    @property
    def shape(self):
        if self._shape is None:
            self.deduce_shape()
        return self._shape
    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def raw_name(self):
        return self.node.name
    @property
    def name(self):
        name = self.node.name
        if name.endswith('bias'):
            name = 'B'
        elif name.endswith('weight'):
            name = 'W'
        elif name.endswith('beta'):
            name = 'beta'
        elif name.endswith('gamma'):
            name = 'gamma'
        elif name.endswith('mean'):
            name = 'mean'
        elif name.endswith('var'):
            name = 'var'
        return name

    @property
    def shape_str(self):
        if self.shape:
            return 'x'.join([str(d) for d in self.shape])
        return ''

    def deduce_shape(self):
        pass
class InOutNode(BaseNode):
    def __init__(self, node):
        super().__init__(node)

    def deduce_shape(self):
        # get type of input tensor
        tensor_type = self.node.type.tensor_type
        # check if it has a shape:
        shape = []
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                elif d.HasField("dim_param"):
                    # unknown dimension with symbolic name
                    shape.append(d.dim_param)
                else:
                    shape.append('?')
        self._shape = shape

class InNode(InOutNode):
    def __init__(self, node):
        super().__init__(node)
        self.op = "Input"

class OutNode(InOutNode):
    def __init__(self, node):
        super().__init__(node)
        self.op = "Output"

class ValueNode(InOutNode):
    def __init__(self, node):
        super().__init__(node)
        self.op = "Value"

class Node(BaseNode):
    def __init__(self, node):
        super().__init__(node)
        self.op = node.op_type
        self.input = node.input
        self.output = node.output
        self.input_blocks = []
        self.output_blocks = []
        self.attributes = _dict()

        self.input_nodes = []
        self.output_nodes = []

        for attrib in node.attribute:
            self.attributes[attrib.name] = self.get_attribute(attrib)

    def get_attribute(self, attribute):
        if attribute.type == onnx.AttributeProto.AttributeType.INT:
            return attribute.i
        elif attribute.type == onnx.AttributeProto.AttributeType.INTS:
            return attribute.ints
        elif attribute.type == onnx.AttributeProto.AttributeType.FLOAT:
            return attribute.f
        elif attribute.type == onnx.AttributeProto.AttributeType.FLOATS:
            return attribute.floats
        elif attribute.type == onnx.AttributeProto.AttributeType.TENSOR:
            return numpy_helper.to_array(attribute.t)
        elif attribute.type == onnx.AttributeProto.AttributeType.TENSORS:
            return numpy_helper.to_array(attribute.tensors)
        elif attribute.type == onnx.AttributeProto.AttributeType.STRING:
            return attribute.s
        elif attribute.type == onnx.AttributeProto.AttributeType.STRINGS:
            return attribute.strings

        assert False
        return None

    def deduce_shape(self):
        node = self.node
        attributes = self.attributes
        inputs = [input.shape for input in self.input_nodes]
        shape = None
        if node.op_type == "Conv":
            X = inputs[0]
            W = inputs[1]
            # todo auto_pad
            stride = attributes.strides
            padding = attributes.pads or [0, 0, 0, 0]
            kernel = attributes.kernel_shape
            shape = [0]*(len(kernel))
            for i in range(len(kernel)):
                shape[i] = int((X[2+i] - kernel[i] + padding[i] + padding[i+len(kernel)])/stride[i]+1)
            shape = [X[0], W[0]] + shape
        elif node.op_type in ['BatchNormalization', 'Relu']:
            shape = inputs[0]
        elif node.op_type in ['MaxPool']:
            input = inputs[0]
            kernel = attributes.kernel_shape
            pads = attributes.pads
            strides = attributes.strides
            dilations = attributes.dilations or [1]*len(kernel)
            shape = [0]*len(kernel)
            ceil_mode = attributes.ceil_mode
            for i in range(len(kernel)):
                sz = (input[2+i] - ((kernel[i]-1)*dilations[i]+1) + pads[i] + pads[i+len(kernel)])/strides[i] + 1
                if ceil_mode:
                    shape[i] = math.ceil(sz)
                else:
                    shape[i] = math.floor(sz)
            shape = input[0:2] + shape
        elif node.op_type in ['Add', 'And', 'Div', 'Equal', 'Greater', 'Less', 'Max', 'Mean', 'Min', 'Mul', 'Or', 'Pow', 'Sub', 'Sum', 'Xor']:
            A = inputs[0]
            B = inputs[1]
            if len(A) < len(B):
                A = [1]*(len(B)-len(A)) + A
            elif len(B) < len(A):
                if attributes.axis is None:
                    B = [1]*(len(A)-len(B)) + B
                else:
                    B = [1]*attributes.axis + B + [1]*(len(A)-len(B)-attributes.axis)
            shape = list(map(max, zip(A, B)))
        elif node.op_type in ['GlobalAveragePool', 'GlobalLpPool', 'GlobalMaxPool']:
            X = inputs[0]
            shape = list(X)
            shape[-2:] = [1, 1]
        elif node.op_type in ['Flatten']:
            input = inputs[0]
            axis = attributes.axis
            if axis is None:
                axis = 1
            shape = [np.prod(input[:axis-1]).astype(int), np.prod(input[axis:]).astype(int)]
        elif node.op_type in ['Reshape']:
            shape = list(attributes.shape)
        elif node.op_type in ['MatMul']:
            A = inputs[0]
            B = inputs[1]
            A_P = []
            if len(A) > 2:
                A_P = A[:-2]
                A = A[-2:]
            B_P = []
            if len(A) > 2:
                B_P = B[:-2]
                B = B[-2:]
            if B_P == A_P:
                B_P = []
            if len(A) == 2 and len(B) == 2:
                shape = [A[0], B[1]]
            elif len(A) == 1:
                shape = B[1:]
            elif len(B) == 1:
                shape = A[:-1]
            shape = A_P + B_P + shape
        elif node.op_type in ['Gemm']:
            A = inputs[0]
            B = inputs[1]
            # C support unidirectional broadcasting
            #C = self.get_shape(inputs[2])
            if attributes.transA:
                A = [A[1], A[0]]
            if attributes.transB:
                B = [B[1], B[0]]
            assert A[1] == B[0]
            shape = [A[0], B[1]]
        self._shape = shape