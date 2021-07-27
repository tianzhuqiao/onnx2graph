import onnx
from graph import Graph

class ONNX2Graph:
    def __init__(self, filename):
        self.model = onnx.load(filename)
        self.graph = Graph(self.model.graph)


if __name__ == "__main__":
    model = ONNX2Graph('resnet18-v1-7.onnx')
    #model = ONNX2Graph('mnist.onnx')
    model.graph.print_graph()
