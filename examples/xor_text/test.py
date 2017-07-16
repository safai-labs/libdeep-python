#!/usr/bin/env python3

import sys
import deeplearn

retval = deeplearn.load("result.nn")
if retval != 0:
    print("Unable to load network. Error code " + str(retval))
    sys.quit()

print("zero,zero"),
if deeplearn.test(["zero","zero"])[0] > 0.5:
    print("1")
else:
    print("0")

print("one,zero"),
if deeplearn.test(["one","zero"])[0] > 0.5:
    print("1")
else:
    print("0")

print("zero,one"),
if deeplearn.test(["zero","one"])[0] > 0.5:
    print("1")
else:
    print("0")

print("one,one"),
if deeplearn.test(["one","one"])[0] > 0.5:
    print("1")
else:
    print("0")
