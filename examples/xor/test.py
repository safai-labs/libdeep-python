#!/usr/bin/python

import sys
import deeplearn

retval = deeplearn.load("result.nn")
if retval != 0:
    print "Unable to load network. Error code " + str(retval)
    sys.quit()

print("0,0"),
if deeplearn.test([0.0,0.0])[0] > 0.5:
    print "1"
else:
    print "0"

print("1,0"),
if deeplearn.test([1.0,0.0])[0] > 0.5:
    print "1"
else:
    print "0"

print("0,1"),
if deeplearn.test([0.0,1.0])[0] > 0.5:
    print "1"
else:
    print "0"

print("1,1"),
if deeplearn.test([1.0,1.0])[0] > 0.5:
    print "1"
else:
    print "0"
