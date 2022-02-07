**onnx2graph** is tool to visualize [onnx](https://onnx.ai/) model with [Graphviz](https://graphviz.org/), for example, to export a model to png, svg, pdf, etc.

### install
1. [install Graphviz](https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt)
2. install this package
    ```
    $ git clone https://github.com/tianzhuqiao/onnx2graph.git
    $ cd onnx2graph
    $ pip install -e .
    ```
### usage
```
$ onnx2graph --help
Usage: onnx2graph [OPTIONS] ONNX

Options:
  --version      Show the version and exit.
  --out TEXT     The output filename, default is same as the onnx file (svg
                 format).
  -v, --verbose
  --help         Show this message and exit.
```

<img src="/doc/mnist.svg"></img>
