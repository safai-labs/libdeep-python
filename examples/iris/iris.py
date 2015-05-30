#!/usr/bin/python

import deeplearn

# The number of iris species to classify
no_of_classes = 3

# Reads a number of data samples from a CSV file
# where the expected output value is the fifth field (index 4)
noOfSamples = deeplearn.readCsvFile("iris.data", 16, 3, [4], no_of_classes)
print str(noOfSamples) + " samples loaded"

# The error threshold (percent) for each layer of the network.
# After going below the threshold the pre-training will move
# on to the next layer
deeplearn.setErrorThresholds([0.5, 0.5, 0.5, 2.5])

# The learning rate in the range 0.0-1.0
deeplearn.setLearningRate(0.1)

# The percentage of dropouts in the range 0-100
deeplearn.setDropoutsPercent(0.01)

# Title of the training error image
deeplearn.setPlotTitle("Iris Species Classification Training")

# The number of time steps after which the training error image is redrawn
deeplearn.setHistoryPlotInterval(500000)

print "Training started"

timeStep = 0
while (deeplearn.training() != 0):
    timeStep = timeStep + 1

print "Training Completed"
print "Test data set performance is " + str(deeplearn.getPerformance()) + "%";

deeplearn.export("result.py")
print "Exported trained network"

deeplearn.save("result.nn")
print "Saved trained network"

deeplearn.free();
print "Done"
