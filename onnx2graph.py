import onnx
from onnx import numpy_helper
import pygraphviz as pgv

from utils import _dict

class ONNX2Graph:
    def __init__(self, filename):
        self.model = onnx.load(filename)
        self.graph = Graph(self.model.graph)

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = _dict()
        self.connections = _dict()

        # update shape info
        self.update_shape(self.graph.input, op='Input')
        self.update_shape(self.graph.output, op='Output')
        self.update_shape(self.graph.value_info)

        self.update_nodes()
        self.update_inout_blocks(self.graph.node)

    def print_graph(self):
        G = pgv.AGraph(directed=True)
        to_nodes = ['Input73']
        done_nodes = {}
        template_table = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{rows}</TABLE>>'''
        template_header = '''<TR><TD BGCOLOR="grey">{name}</TD></TR>'''
        template_row = '''<TR><TD ALIGN="left">{name}</TD></TR>'''
        while to_nodes:
            name = to_nodes[0]
            done_nodes[to_nodes[0]] = True
            node = self.nodes[to_nodes.pop(0)]
            print(f'{node.op}({name})')

            if 'output_blocks' not in node:
                node.output_blocks = []
            for out in node.output_blocks:
                if out not in done_nodes:
                    to_nodes.append(out)
            if name in self.connections:
                for out in self.connections[name].outputs:
                    if out not in done_nodes:
                        to_nodes.append(out)
            if 'input' not in node:
                node.input = []
            inputs = []
            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                inputs.append(template_row.format(name=f'In: {input[0]} ({self.nodes[i].shape})'))
                print(f'   in: {input} {self.nodes[i].shape}')
            if not node.output:
                node.output = []
            outputs = []
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                outputs.append(template_row.format(name=f'Out:{output[0]} ({self.nodes[o].shape})'))
                print(f'   out:{output} {self.nodes[o].shape}')
            header = template_header.format(name=f'{node.op} ({name})')
            rows = "\n".join([header] + inputs + outputs)
            G.add_node(f'{name}', label=template_table.format(rows=rows), shape='plaintext')

        all_nodes = done_nodes
        # add edges
        to_nodes = ['Input73']
        done_nodes = {}
        while to_nodes:
            name = to_nodes[0]
            done_nodes[to_nodes[0]] = True
            node = self.nodes[to_nodes.pop(0)]

            for out in node.output_blocks:
                if out not in done_nodes:
                    to_nodes.append(out)
            if name in self.connections:
                for out in self.connections[name].outputs:
                    if out not in done_nodes:
                        to_nodes.append(out)

            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                if input[0] in all_nodes:
                    G.add_edge(input[0], name)

            outputs = []
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                if output[0] in all_nodes:
                    G.add_edge(name, output[0])

        G.layout(prog='dot')
        G.write("mnist.dot")
        G.draw("mnist.png")

    def print_summary(self):
        to_nodes = ['Input73']
        done_nodes = {}
        while to_nodes:
            name = to_nodes[0]
            done_nodes[to_nodes[0]] = True
            node = self.nodes[to_nodes.pop(0)]
            print(f'{node.op}({name})')
            if 'output_blocks' not in node:
                node.output_blocks = []
            for out in node.output_blocks:
                if out not in done_nodes:
                    to_nodes.append(out)
            if name in self.connections:
                for out in self.connections[name].outputs:
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

    def update_nodes(self):
        for node in self.graph.node:
            if node.name not in self.nodes:
                self.nodes[node.name] = _dict(name=node.name)
            self.nodes[node.name].update({'op': node.op_type, 'input': node.input, 'output': node.output})
            for i in node.input:
                if i not in self.connections:
                    self.connections[i] = _dict({'outputs': [], 'inputs': []})
                self.connections[i].outputs.append(node.name)
            for o in node.output:
                if o not in self.connections:
                    self.connections[o] = _dict({'outputs': [], 'inputs': []})
                self.connections[o].inputs.append(node.name)

    def update_inout_blocks(self, nodes):
        for node in nodes:
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

    def update_shape(self, nodes, **kwargs):
        for node in nodes:
            if node.name not in self.nodes:
                self.nodes[node.name] = _dict(name=node.name)
            self.nodes[node.name].update(kwargs)
            self.nodes[node.name].shape = self.get_node_shape(node)

    def get_node_shape(self, node):
        # get type of input tensor
        tensor_type = node.type.tensor_type
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

    def get_constant_value(self, const):
        values = []
        for attri in const.attribute:
            val = numpy_helper.to_array(attri.t)
            values.append(val)
        if len(values) == 1:
            return values[0]
        return values

if __name__ == "__main__":
    model = ONNX2Graph('mnist.onnx')
    model.graph.print_graph()
