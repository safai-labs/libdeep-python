# libdeep-python
A python interface for the libdeep deep learning library

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

Enter Python's interactive console by typing "python", then import the deep learning module:

````
import deeplearn
````

To create a neural net with 20 inputs, 10 hidden units per layer, 3 hidden layers and 5 outputs:

````
deeplearn.init(20,10,3,5)
````
