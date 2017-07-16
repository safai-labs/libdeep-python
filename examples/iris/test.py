#!/usr/bin/env python3

import deeplearn

# Rather than showing numbers show the species names

species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Load the neural network

deeplearn.load("result.nn")

# These dimensions are similar to those which exist in the
# data set, but are adjusted slightly so that the network
# has never seen these exact values before

print("Expected: " + species[0])
deeplearn.test([5.44, 3.436, 1.667, 0.214])
print("Returned: " + species[deeplearn.getClass()])

print("\nExpected: " + species[1])
deeplearn.test([6.14, 2.75, 4.04, 1.32])
print("Returned: " + species[deeplearn.getClass()])

print("\nExpected: " + species[2])
deeplearn.test([6.71, 3.14, 5.92, 2.29])
print("Returned: " + species[deeplearn.getClass()])
