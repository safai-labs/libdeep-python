# libdeep-python

<img src="https://github.com/bashrc/libdeep-python/blob/master/img/trainingexample.jpg?raw=true" width=640/>

libdeep-python adds a Python API to the [libdeep](https://github.com/bashrc/libdeep) deep learning library, which is written in C. This allows you to obtain the raw processing speed of native C code while also having the convenience of being able to develop your application in Python. After a system has been trained you can export it as a standalone C commandline program which takes the input values as arguments and outputs the results to stdout.

Installation
------------

To install dependencies on a Debian based system:

``` bash
sudo apt-get install build-essential doxygen python3-dev
```

Or on an Arch based system:

``` bash
sudo pacman -S gcc doxygen
```

Then install libdeep:

``` bash
git clone https://github.com/bashrc/libdeep
cd libdeep
make
sudo make install
```

Finally install the python interface:

``` bash
git clone https://github.com/bashrc/libdeep-python
cd libdeep-python
make
sudo make install
```

Usage
-----

For example use cases see the examples directory and also the manpage.

    man libdeep-python

Try an Example
--------------

To check that the system is working you can try the simplest example, which is learning the XOR function.

``` bash
cd examples/xor
./xor.py
```
