from distutils.core import setup, Extension

# define the extension module
deeplearn_module = Extension('deeplearn', sources=['src/deeplearn.c'], libraries=['deep'])
# run the setup
setup(ext_modules=[deeplearn_module])
