import os
import codecs
from setuptools import setup

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
    raise RuntimeError("Unable to open file %s"%rel_path)

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(name='onnx2graph',
      version=get_version('onnx2graph/__init__.py'),
      description='ONNX to graph',
      author='Tianzhu Qiao',
      author_email='tq@feiyilin.com',
      license="MIT",
      platforms=["any"],
      py_modules=['onnx2graph'],
      install_requires=['onnx', 'click'],
      entry_points='''
        [console_scripts]
        onnx2graph=onnx2graph.__main__:cli
      '''
     )
