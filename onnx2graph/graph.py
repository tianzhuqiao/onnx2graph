import click
import numpy as np
import html
from onnx import numpy_helper
import pygraphviz as pgv

from .utils import _dict
from .node import Node, InNode, OutNode, ValueNode

class Graph:
    template_table = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{rows}</TABLE>>'''
    template_header = '''<TR><TD BGCOLOR="grey" ALIGN="left">{name}</TD></TR>'''
    template_row = '''<TR><TD ALIGN="left">{name}</TD></TR>'''

    def __init__(self, model, shape=False,verbose=False):
        self.verbose = verbose
        self.decuce_shape = shape
        self.graph = model.graph
        self.model = model
        self.op_version = model.opset_import[0].version
        Node.op_version = self.op_version
        self.nodes = _dict()
        self.connections = _dict()

        self.add_all_nodes()

        self.update_nodes()
        self.update_inout_blocks(self.graph.node)

    def echo(self, msg):
        if self.verbose:
            click.echo(msg)

    def get_all_nodes(self):
        initializers = [n.name for n in self.graph.initializer]
        # get the "required" inputs
        inputs = [n.name for n in self.graph.input if n.name not in initializers]

        nodes = [n.name for n in self.graph.node if n.op_type != 'Constant']
        outputs = [f'{n.name}' for n in self.graph.output]
        return (inputs + nodes + outputs, inputs, nodes, outputs)

    def print_input_node(self, G):
        initializers = [n.name for n in self.graph.initializer]
        # get the "required" inputs
        inputs = [n.name for n in self.graph.input if n.name not in initializers]
        for nd in inputs:
            name = nd
            node = self.nodes[name]
            self.echo(f'{node.op}({name})')

            header = self.template_header.format(name=f'{node.op}')
            rows = "\n".join([header])
            G.add_node(f'i-{name}', label=self.template_table.format(rows=rows), shape='plaintext')

    def print_output_node(self, G):
        outputs = [n.name for n in self.graph.output]
        for nd in outputs:
            name = nd
            node = self.nodes[name]
            self.echo(f'{node.op}({name})')

            header = self.template_header.format(name=f'{node.op}')
            rows = "\n".join([header])
            G.add_node(f'o-{name}', label=self.template_table.format(rows=rows), shape='plaintext')

    def print_nodes(self, G):
        all_nodes, _, nodes, _ = self.get_all_nodes()
        for nd in nodes:
            name = nd
            node = self.nodes[name]
            self.echo(f'{node.op}({name})')

            inputs = []
            for idx, i in enumerate(node.input):
                input = [i]
                label = node.input_label(idx)
                notes = node.input_notes(idx)
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                if input[0] not in all_nodes:
                    # not node, may be constant/values/weights ...
                    inputs.append(self.template_row.format(name=f'<b>{html.escape(label)}</b> &nbsp; {html.escape(notes)}'))
                self.echo(f'   in: {label} {self.nodes[input[0]].shape}')

            outputs = []
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                if output[0] not in all_nodes and output[0] in self.nodes:
                    out_name = self.nodes[output[0]].name
                    outputs.append(self.template_row.format(name=f'Out:{out_name} ({self.nodes[o].shape_str})'))
                    self.echo(f'   out:{output} {self.nodes[output[0]].shape}')
            header = self.template_header.format(name=f'{node.op}')
            rows = "\n".join([header] + inputs + outputs)
            G.add_node(f'{name}', label=self.template_table.format(rows=rows), shape='plaintext')

    def print_edge(self, G):
        # add edges
        all_nodes, all_inputs, nodes, all_outputs = self.get_all_nodes()
        for nd in nodes:
            name = nd
            node = self.nodes[name]

            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                if input[0] in all_nodes:
                    in_name = input[0]
                    if in_name in all_inputs:
                        in_name = f'i-{input[0]}'
                    if self.decuce_shape and self.nodes[input[0]].shape:
                        G.add_edge(in_name, name, label=self.nodes[input[0]].shape_str)
                    else:
                        G.add_edge(in_name, name)

            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                if output[0] in all_nodes:
                    if ((output[0] not in nodes) or (output[0] == name)) and (output[0] in all_outputs):
                        # some onnx may have same name in graph.output and graph.node
                        output[0] = f'o-{output[0]}'
                    if self.decuce_shape and node.shape:
                        G.add_edge(name, output[0], label=node.shape_str)
                    else:
                        G.add_edge(name, output[0])

    def print_graph(self, out):
        G = pgv.AGraph(directed=True)

        # print nodes
        self.print_input_node(G)
        self.print_nodes(G)
        self.print_output_node(G)

        self.print_edge(G)

        G.layout(prog='dot')
        G.draw(out)

    def update_nodes(self):
        for node in self.graph.node:
            assert node.name in self.nodes
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
            input_nodes = []
            for i in node.input:
                input = [i]
                if i in self.connections and self.connections[i].inputs:
                    input = self.connections[i].inputs
                self.nodes[node.name].input_blocks += input
                input_nodes.append(self.nodes[input[0]])
            self.nodes[node.name].input_nodes = input_nodes
            output_nodes = []
            for o in node.output:
                output = [o]
                if o in self.connections and self.connections[o].outputs:
                    output = self.connections[o].outputs
                self.nodes[node.name].output_blocks += output
                output_nodes.append(self.nodes.get(output[0], None))
            self.nodes[node.name].output_nodes = output_nodes

    def add_node(self, node, ntype):
        if not node.name:
            node.name = "_".join([node.op_type]+list(node.input)+list(node.output))
        nd = None
        if ntype == "input":
            nd = InNode(node)
        elif ntype == "output":
            nd = OutNode(node)
        elif ntype == "value_info":
            nd = ValueNode(node)
        else:
            nd = Node(node)
        if node.name in self.nodes:
            #if self.nodes[node.name].shape is None:
            self.nodes[node.name].shape = nd.shape
            assert not isinstance(nd, Node)
        else:
            self.nodes[node.name] = nd

    def add_all_nodes(self):
        # list all input, output and node from the graph
        def add_nodes(nodes, ntype=''):
            for node in nodes:
                self.add_node(node, ntype)
        add_nodes(self.graph.node)
        add_nodes(self.graph.input, 'input')
        add_nodes(self.graph.output, 'output')
        add_nodes(self.graph.value_info, 'value_info')
        for val in self.graph.initializer:
            if val.name in self.nodes:
                self.nodes[val.name].values = numpy_helper.to_array(val)

    def get_shape(self, name):
        shape = self.nodes[name].shape
        if not shape and name in self.connections:
            input = self.connections[name]['inputs'][0]
            shape = self.nodes[input].shape
        return list(shape)

    def get_constant_value(self, const):
        values = []
        for attri in const.attribute:
            val = numpy_helper.to_array(attri.t)
            values.append(val)
        if len(values) == 1:
            return values[0]
        return values
