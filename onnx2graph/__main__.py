import os
import click
import onnx
from onnx2graph.graph import Graph
from onnx2graph import __version__

class ONNX2Graph:
    def __init__(self, filename, *args, **kwargs):
        self.model = onnx.load(filename)
        self.graph = Graph(self.model, *args, **kwargs)

@click.command()
@click.version_option(__version__)
@click.argument('onnx', type=click.Path(dir_okay=False, readable=True))
@click.option('--out', help='The output filename, default is same as the onnx file (svg format).')
@click.option('--shape', is_flag=True, help='Deduce the shape of each block if possible (experimental).')
@click.option('-v', '--verbose', count=True)
def cli(onnx, out, shape, verbose):
    model = ONNX2Graph(onnx, shape, verbose)
    if not out:
        out = f'{os.path.splitext(onnx)[0]}.svg'
    model.graph.print_graph(out)

if __name__ == "__main__":
    cli()
