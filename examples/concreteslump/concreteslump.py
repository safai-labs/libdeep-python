#!/usr/bin/python

import deeplearn

# Reads a number of data samples from a CSV file
# where the expected output values are within field indexes 7,8,9 and 10
noOfSamples = deeplearn.readCsvFile("slump_test.data", 16, 3, [7,8,9,10], 0)
print str(noOfSamples) + " samples loaded"

# The error threshold (percent) for each layer of the network.
# Three hidden layers, plus the final training.
# After going below the threshold the pre-training will move
# on to the next layer
deeplearn.setErrorThresholds([0.5, 0.5, 0.5, 2.0])

# The learning rate in the range 0.0-1.0
deeplearn.setLearningRate(0.2)

# The percentage of dropouts in the range 0-100
deeplearn.setDropoutsPercent(0.001)

# Title of the training error image
deeplearn.setPlotTitle("Concrete Slump Training")

# The number of time steps after which the training error image is redrawn
deeplearn.setHistoryPlotInterval(900000)

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
