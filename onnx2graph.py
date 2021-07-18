import onnx
from onnx import numpy_helper

class _dict(dict):
    """dict like object that exposes keys as attributes"""
    def __getattr__(self, key):
        ret = self.get(key)
        if not ret and key.startswith("__"):
            raise AttributeError()
        return ret
    def __setattr__(self, key, value):
        self[key] = value
    def __getstate__(self):
        return self
    def __setstate__(self, d):
        self.update(d)
    def update(self, d):
        """update and return self -- the missing dict feature in python"""
        super(_dict, self).update(d)
        return self
    def copy(self):
        return _dict(dict(self).copy())

class ONNX2Graph:
    def __init__(self, filename):
        self.model = onnx.load(filename)
        self.inputs = _dict()
        self.nodes = _dict()
        for node in self.model.graph.input:
            self.inputs[node.name] = _dict({'shape': self.get_node_shape(node)})
            self.nodes[node.name] = _dict({'shape': self.get_node_shape(node)})
        self.outputs = {}
        for node in self.model.graph.output:
            self.outputs[node.name] = _dict({'shape': self.get_node_shape(node)})
            self.nodes[node.name] = _dict({'shape': self.get_node_shape(node), 'output_blocks':[], 'op': 'Output'})

        for node in self.model.graph.value_info:
            self.nodes[node.name] = _dict({'shape': self.get_node_shape(node)})

        self.constants = _dict()
        self.connections = _dict()
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                self.constants[node.name] = self.get_constant_value(node)
            if node.name not in self.nodes:
                self.nodes[node.name] = _dict()
            self.nodes[node.name].update({'op': node.op_type, 'input': node.input, 'output': node.output})
            for i in node.input:
                if i not in self.connections:
                    self.connections[i] = _dict({'outputs': [], 'inputs': []})
                self.connections[i].outputs.append(node.name)
            for o in node.output:
                if o not in self.connections:
                    self.connections[o] = _dict({'outputs': [], 'inputs': []})
                self.connections[o].inputs.append(node.name)

        for node in self.model.graph.node:
            #if node.op_type in ['Reshape', 'Add']:
            #    continue
            if node.op_type != 'Constant':
                print(f'{node.op_type:<10}', end='')
                for i in node.input:
                    if i in self.inputs:
                        print(f'{"x".join(str(d) for d in self.inputs[i]["shape"]):<10}', end='')
                print(f'{node.output} {self.nodes[node.output[0]]} {node.input} self.nodes[node.input[0]]')
                print('')
            self.nodes[node.name].input_blocks = []
            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                self.nodes[node.name].input_blocks += input
            self.nodes[node.name].output_blocks = []
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                self.nodes[node.name].output_blocks += output
        print('=======')
        to_nodes = ['Block386']
        done_nodes = {}
        while to_nodes:
            name = to_nodes[0]
            done_nodes[to_nodes[0]] = True
            node = self.nodes[to_nodes.pop(0)]
            print(f'{node.op}({name})')
            for out in node.output_blocks:
                if out not in done_nodes:
                    to_nodes.append(out)
            if 'input' not in node:
                node.input = []
            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                print(f'   in: {input} {self.nodes[i].shape}')
            if not node.output:
                continue
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                print(f'   out:{output} {self.nodes[o].shape}')


    def get_node_shape(self, node):
        # get type of input tensor
        tensor_type = node.type.tensor_type
        # check if it has a shape:
        shape = []
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    shape.append(d.dim_value)
                elif (d.HasField("dim_param")):
                    # unknown dimension with symbolic name
                    shape.append(d.dim_param)
                else:
                    shape.append('?')
        return shape

    def get_constant_value(self, const):
        values = []
        for attri in const.attribute:
            val = numpy_helper.to_array(attri.t)
            values.append(val)
        if len(values) == 1:
            return values[0]
        return values


model = ONNX2Graph('mnist.onnx')
#for node in model.graph.node:
#  print(node.name, node.type)
