import onnx
from onnx import numpy_helper

from utils import _dict
class BaseNode:
    def __init__(self, node):
        self.node = node
        self.shape = None
        self.input = []
        self.output = []
        self.input_blocks = []
        self.output_blocks = []
        self.attributes = _dict()

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

class InOutNode(BaseNode):
    def __init__(self, node):
        super().__init__(node)
        self.shape = self.get_shape()

    def get_shape(self):
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
        return shape

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

        for attrib in node.attribute:
            self.attributes[attrib.name] = self.get_attribute(attrib)

        self.shape = None

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
