# libdeep-python
libdeep-python adds a Python API to the libdeep deep learning library, which is written in C. This allows you to obtain the raw processing speed of native C code while also having the convenience of being able to develop your application in Python. You'll need to import the "deeplearn" library. After a system has been trained you can export it as a standalone C commandline program which takes the input values as arguments and outputs the results to stdout.

Installation
------------

First install libdeep:

````
sudo apt-get install build-essential gnuplot doxygen
git clone https://github.com/bashrc/libdeep
cd libdeep
make
sudo make install
````

Now install the python interface:

````
sudo apt-get install python-dev
git clone https://github.com/bashrc/libdeep-python
cd libdeep-python
make
sudo make install
````

Usage
-----

For example use cases see the examples directory.
