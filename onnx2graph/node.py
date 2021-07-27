import math
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import defs
from utils import _dict

schemas = _dict()
for schema in defs.get_all_schemas_with_history():
    if schema.name not in schemas:
        schemas[schema.name] = [schema]
    else:
        schemas[schema.name].append(schema)


class BaseNode:
    def __init__(self, node):
        self.node = node
        self.input = []
        self.output = []
        self.input_blocks = []
        self.output_blocks = []
        self.attributes = _dict()
        self._shape = None
        self.op = ""
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

        self._input_nodes = []
        self._output_nodes = []

        for attrib in node.attribute:
            self.attributes[attrib.name] = self.get_attribute(attrib)

    @property
    def input_nodes(self):
        return self._input_nodes

    @input_nodes.setter
    def input_nodes(self, nodes):
        self._input_nodes = nodes

    def input_label(self, index):
        if index < 0 or index > len(self.input):
            return None
        op = schemas.get(self.op, None)
        if not op:
            return self.input[index]
        return op[0].inputs[index].name

    def input_notes(self, index):
        if self.shape is not None:
            if self.shape:
                return f'<{self.shape_str}>'
            if self.op == 'Constant':
                value = None
                for v in ['value', 'value_float', 'value_int', 'value_string']:
                    if v in self.attributes:
                        value = self.attributes[v]
                        break
                return f'({value})'
        return ''

    @property
    def output_nodes(self):
        return self._output_nodes
    @output_nodes.setter
    def output_nodes(self, nodes):
        self._output_nodes = nodes

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

    def deduce_shape(self, index=0):
        attributes = self.attributes
        inputs = [list(input.shape) for input in self.input_nodes]
        shape = None
        op = self.op
        if op in ["Conv", 'ConvInteger']:
            X = inputs[0]
            W = inputs[1]
            # todo auto_pad
            stride = attributes.strides
            kernel = attributes.kernel_shape
            padding = attributes.pads or [0]*(len(kernel)*2)
            dilations = attributes.dilations or [1]*(len(kernel))
            shape = [0]*(len(kernel))
            for i in range(len(kernel)):
                shape[i] = int((X[2+i] -  ((kernel[i]-1)*dilations[i]+1) + padding[i] + padding[i+len(kernel)])/stride[i]+1)
            shape = [X[0], W[0]] + shape
        elif op in ['ConvTranspose']:
            X = inputs[0]
            W = inputs[1]
            # todo auto_pad
            output_shape = attributes.output_shape
            stride = attributes.strides
            kernel = attributes.kernel_shape
            output_padding = attributes.output_padding or [0]*(len(kernel)*2)
            pads = attributes.pads or [0]*(len(kernel)*2)
            dilations = attributes.dilations or [1]*(len(kernel))
            group = attributes.group
            if output_shape:
                shape = output_shape
            else:
                shape = [0]*(len(kernel))
                for i in range(len(kernel)):
                    shape[i] = stride[i] * (X[2+i] - 1) + output_padding[i] + ((kernel[i] - 1) * dilations[i] + 1) - pads[i] - pads[i+len(kernel)]
                shape = [X[0], W[0]*group] + shape
        elif op in ['BatchNormalization', 'Relu', 'Abs', 'Acos', 'Acosh', 'Asin',
                    'Asinh', 'Atan', 'Atanh', 'Cast', 'CastLike', 'Ceil', 'Celu',
                    'Clip', 'Cos', 'Cosh', 'CumSum', 'DequantizeLinear', 'Dropout',
                    'Elu', 'Erf', 'Exp', 'EyeLike', 'Floor']:
            shape = inputs[0]
        elif op in ['MaxPool', 'AveragePool']:
            input = inputs[0]
            kernel = attributes.kernel_shape
            pads = attributes.pads
            strides = attributes.strides
            dilations = attributes.dilations or [1]*len(kernel)
            ceil_mode = attributes.ceil_mode

            shape = [0]*len(kernel)
            for i in range(len(kernel)):
                sz = (input[2+i] - ((kernel[i]-1)*dilations[i]+1) + pads[i] + pads[i+len(kernel)])/strides[i] + 1
                if ceil_mode:
                    shape[i] = math.ceil(sz)
                else:
                    shape[i] = math.floor(sz)
            shape = input[0:2] + shape
        elif op in ['Add', 'And', 'Div', 'Equal', 'Greater', 'Less', 'Max', 'Mean', 'Min', 'Mul', 'Or', 'Pow', 'Sub', 'Sum', 'Xor', 'BitShift']:
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
        elif op in ['GlobalAveragePool', 'GlobalLpPool', 'GlobalMaxPool']:
            X = inputs[0]
            shape = list(X)
            shape[-2:] = [1, 1]
        elif op in ['Flatten']:
            input = inputs[0]
            axis = attributes.axis
            if axis is None:
                axis = 1
            shape = [np.prod(input[:axis-1]).astype(int), np.prod(input[axis:]).astype(int)]
        elif op in ['Reshape']:
            shape = list(attributes.shape)
        elif op in ['MatMul']:
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
        elif op in ['Gemm']:
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
        elif op in ['ArgMax', 'ArgMin']:
            input = inputs[0]
            axis = attributes.get('axis', 0)
            keepdims = attributes.get('keepdims', 1)
            # select_last_index = attributes.get('select_last_index', False)
            shape = input
            shape[axis] = 1
            if not keepdims:
                shape = shape[:keepdims] + shape[keepdims+1:]
        elif op in ['Compress', 'ConstantOfShape']:
            # the actual size may depends on the data
            shape = None
        elif op in ['Concat']:
            axis = attributes.axis
            shape = inputs[0]
            for input in inputs[1:]:
                shape[axis] += input[axis]
        elif op in ['Constant']:
            for v in ['sparse_value', 'value']:
                if v in attributes:
                    shape = list(attributes[v].shape)
            for v in ['value_float', 'value_int', 'value_string']:
                if v in attributes:
                    shape = []
            for v in ['value_floats', 'value_ints', 'value_strings']:
                if v in attributes:
                    shape = [len(attributes[v])]
        elif op in ['DepthToSpace']:
            X = inputs[0]
            blocksize = attributes.blocksize
            shape = list(X)
            shape[1] = shape[1]//(blocksize**2)
            shape[2] *= blocksize
            shape[3] *= blocksize
        elif op in ['Det']:
            X = inputs[0]
            shape = X[:-2]
        elif op in ['Gru']:
            X = inputs[0]
            W = inputs[1]
            if index == 0:
                shape = [X[0], W[0], X[1], W[1]/3]
            elif index == 1:
                shape = [W[0], X[1], W[1]/3]
        elif op in ['Gather']:
            axis = attributes.axis or 0
            data = inputs[0]
            indices = inputs[1]

            shape = data[:axis] + data[axis+1:] + indices
        elif op in ['GatherElements']:
            indices = inputs[1]
            shape = indices
        elif op in ['GatherND']:
            pass
        self._shape = shape
