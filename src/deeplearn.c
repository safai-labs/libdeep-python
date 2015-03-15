/*
 libdeep-python - Python interface for libdeep
 Copyright (C) 2015  Bob Mottram <bob@robotics.uk.to>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the University nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
 .
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE HOLDERS OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <Python.h>
#include <libdeep/globals.h>
#include <libdeep/deeplearn.h>
#include <libdeep/deeplearndata.h>

static int initialised = 0;
static deeplearn learner;
static unsigned int random_seed = 46362;
static float error_threshold[256];

static PyObject* getErrorThreshold(PyObject* self, PyObject* args)
{
    int index=0;
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "i", &index))
        return Py_BuildValue("i", -2);
    return Py_BuildValue("f", deeplearn_get_error_threshold(&learner,index));
}

static PyObject* currentLayer(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", learner.current_hidden_layer);
}

static PyObject* backpropError(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("f", learner.BPerror);
}

static PyObject* inputs(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", learner.net->NoOfInputs);
}

static PyObject* outputs(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", learner.net->NoOfOutputs);
}

static PyObject* hiddens(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", learner.net->NoOfHiddens);
}

static PyObject* layers(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", learner.net->HiddenLayers);
}

static PyObject* setErrorThresholds(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int index = 0;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return Py_BuildValue("i", -1);
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -2);
    }

    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (index >= 20) {
            return Py_BuildValue("i", -3);
        }

        if (!PyFloat_Check(next)) {
            /* error, we were expecting a floating point value */
            return Py_BuildValue("i", -4);
        }

        deeplearn_set_error_threshold(&learner, index, (float)PyFloat_AsDouble(next));
        index++;
    }

    return Py_BuildValue("i", 0);
}

/* sets an error threshold for training a layer */
static PyObject* setErrorThreshold(PyObject* self, PyObject* args)
{
    int layer_index=0;
    float threshold=0.1f;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "if", &layer_index, &threshold))
        return Py_BuildValue("i", -2);
    deeplearn_set_error_threshold(&learner, layer_index, threshold);
    return Py_BuildValue("i", 0);
}

static PyObject* setLearningRate(PyObject* self, PyObject* args)
{
    float rate=0.1f;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "f", &rate))
        return Py_BuildValue("i", -2);
    deeplearn_set_learning_rate(&learner, rate);
    return Py_BuildValue("i", 0);
}

static PyObject* setDropoutsPercent(PyObject* self, PyObject* args)
{
    float percent=2.0f;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "f", &percent))
        return Py_BuildValue("i", -2);
    deeplearn_set_dropouts(&learner, percent);
    return Py_BuildValue("i", 0);
}

/* set the random seed */
static PyObject* setSeed(PyObject* self, PyObject* args)
{
    int randseed=0;

    if (!PyArg_ParseTuple(args, "i", \
                          &randseed))
        return NULL;
    random_seed = (unsigned int)randseed;
    return Py_BuildValue("i", 0);
}

static PyObject* init(PyObject* self, PyObject* args)
{
    int no_of_inputs=0;
    int no_of_hiddens=0;
    int hidden_layers=0;
    int no_of_outputs=0;

    if (initialised != 0) {
        deeplearn_free(&learner);
        initialised = 0;
    }

    if (!PyArg_ParseTuple(args, "iiii",
                          &no_of_inputs,
                          &no_of_hiddens,
                          &hidden_layers,
                          &no_of_outputs))
        return Py_BuildValue("i", -1);

    deeplearn_init(&learner,
                   no_of_inputs,
                   no_of_hiddens,
                   hidden_layers,
                   no_of_outputs,
                   error_threshold,
                   &random_seed);

    initialised = 1;
    return Py_BuildValue("i", 0);
}

static PyObject* feedForward(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    deeplearn_feed_forward(&learner);
    return Py_BuildValue("i", 0);
}

static PyObject* update(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    deeplearn_update(&learner);
    return Py_BuildValue("i", 0);
}

static PyObject* setInput(PyObject* self, PyObject* args)
{
    int index=0;
    float value=0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "if", &index, &value))
        return Py_BuildValue("i", -2);

    deeplearn_set_input(&learner, index, value);
    return Py_BuildValue("i", 0);
}


static PyObject* setInputs(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int index = 0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return Py_BuildValue("i", -2);
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -3);
    }

    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (index >= learner.net->NoOfInputs) {
            return Py_BuildValue("i", -4);
        }

        if (!PyFloat_Check(next)) {
            /* error, we were expecting a floating point value */
            return Py_BuildValue("i", -5);
        }

        double value = PyFloat_AsDouble(next);
        deeplearn_set_input(&learner, index, (float)value);
        index++;
    }
    if (index != learner.net->NoOfInputs) {
        return Py_BuildValue("i", -6);
    }

    return Py_BuildValue("i", 0);
}

static PyObject* setFields(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int index = 0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return Py_BuildValue("i", -2);
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -3);
    }

    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (index >= learner.no_of_input_fields) {
            return Py_BuildValue("i", -4);
        }

        if (learner.field_length[index] == 0) {
            if (!PyFloat_Check(next)) {
                /* error, we were expecting a floating point value */
                return Py_BuildValue("i", -5);
            }
            double value = PyFloat_AsDouble(next);
            deeplearn_set_input_field(&learner, index, (float)value);
        }
        else {
            char * text = PyString_AsString(next);
            deeplearn_set_input_field_text(&learner, index, text);
        }

        index++;
    }
    if (index != learner.no_of_input_fields) {
        return Py_BuildValue("i", -6);
    }

    return Py_BuildValue("i", 0);
}

static PyObject* test(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int index = 0;
    PyObject *pylist, *item;
    float value, range, normalised;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return Py_BuildValue("i", -2);
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -3);
    }

    /* read the inputs list */
    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (index >= learner.net->NoOfInputs) {
            return Py_BuildValue("i", -4);
        }

        if (!PyFloat_Check(next)) {
            /* error, we were expecting a floating point value */
            return Py_BuildValue("i", -5);
        }

        value = (float)PyFloat_AsDouble(next);
        range = learner.input_range_max[index] - learner.input_range_min[index];
        if (range > 0.001f) {
            normalised = (((value - learner.input_range_min[index])/range)*0.5f) + 0.25f;
            if (normalised < 0.25f) {
                normalised = 0.25f;
            }
            if (normalised > 0.75f) {
                normalised = 0.75f;
            }
            deeplearn_set_input(&learner, index, normalised);
        }
        index++;
    }
    if (index != learner.net->NoOfInputs) {
        return Py_BuildValue("i", -6);
    }

    /* update the network */
    deeplearn_feed_forward(&learner);

    /* return the outputs as a list */
    pylist = PyList_New(learner.net->NoOfOutputs);
    for (index = 0; index < learner.net->NoOfOutputs; index++) {
        value = deeplearn_get_output(&learner, index);
        range = learner.output_range_max[index] - learner.output_range_min[index];
        normalised = -9999;
        if (range > 0.001f) {
            normalised = (((value - 0.25f)/0.5f)*range) + learner.output_range_min[index];
        }
        item = PyFloat_FromDouble((double)normalised);
        PyList_SET_ITEM(pylist, index, item);
    }

    return pylist;
}

static PyObject* setOutput(PyObject* self, PyObject* args)
{
    int index=0;
    float value=0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "if", &index, &value))
        return Py_BuildValue("i", -2);

    deeplearn_set_output(&learner, index, value);
    return Py_BuildValue("i", 0);
}

static PyObject* setHistoryPlotInterval(PyObject* self, PyObject* args)
{
    int interval=0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "i", &interval))
        return Py_BuildValue("i", -2);

    learner.history_plot_interval = interval;
    return Py_BuildValue("i", 0);
}

static PyObject* setPlotTitle(PyObject* self, PyObject* args)
{
    char * title;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "s", &title))
        return Py_BuildValue("i", -2);

    sprintf(learner.history_plot_title,"%s",title);
    return Py_BuildValue("i", 0);
}

static PyObject* readCsvFile(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int retval;
    char * filename;
    int no_of_hiddens, hidden_layers, no_of_outputs=0, output_classes=0;
    int output_field_index[256];

    if (!PyArg_ParseTuple(args, "siiOi", &filename, &no_of_hiddens, &hidden_layers, &obj, &output_classes))
        return Py_BuildValue("i", -1);

    initialised = 1;

    /* get the field indexes of outputs */
    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -2);
    }

    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (!PyInt_Check(next)) {
            /* error, we were expecting an integer value */
            return Py_BuildValue("i", -4);
        }

        output_field_index[no_of_outputs++] = (int)PyInt_AsLong(next);
    }

    if (no_of_outputs == 0) {
        return Py_BuildValue("i", -5);
    }

    retval = deeplearndata_read_csv(filename, &learner,
                                    no_of_hiddens, hidden_layers,
                                    no_of_outputs,
                                    output_field_index,
                                    output_classes,
                                    error_threshold,
                                    &random_seed);
    return Py_BuildValue("i", retval);
}

static PyObject* getOutput(PyObject* self, PyObject* args)
{
    int index=0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "i", &index))
        return Py_BuildValue("i", -2);

    if (index < 0) {
        return Py_BuildValue("i", -3);
    }

    if (index >= learner.net->NoOfOutputs) {
        return Py_BuildValue("i", -4);
    }

    return Py_BuildValue("f", deeplearn_get_output(&learner, index));
}

static PyObject* getClass(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    return Py_BuildValue("i", deeplearn_get_class(&learner));
}

static PyObject* setClass(PyObject* self, PyObject* args)
{
    int class = 0;
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "i", &class)) {
        return Py_BuildValue("i", -2);
    }

    deeplearn_set_class(&learner, class);
    return Py_BuildValue("i", 0);
}

static PyObject* setOutputs(PyObject* self, PyObject* args)
{
    PyObject *obj;
    int index = 0;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return Py_BuildValue("i", -2);
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return Py_BuildValue("i", -3);
    }

    while (1) {
        PyObject *next = PyIter_Next(iter);
        if (!next) {
            /* nothing left in the iterator */
            break;
        }

        if (index >= learner.net->NoOfOutputs) {
            return Py_BuildValue("i", -4);
        }

        if (!PyFloat_Check(next)) {
            /* error, we were expecting a floating point value */
            return Py_BuildValue("i", -5);
        }

        double value = PyFloat_AsDouble(next);
        deeplearn_set_output(&learner, index, (float)value);
        index++;
    }
    if (index != learner.net->NoOfOutputs) {
        return Py_BuildValue("i", -6);
    }

    return Py_BuildValue("i", 0);
}

static PyObject* save(PyObject* self, PyObject* args)
{
    FILE * fp;
    char * filename;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }
    if (!PyArg_ParseTuple(args, "s", &filename))
        return Py_BuildValue("i", -2);

    fp = fopen(filename,"w");
    if (fp) {
        deeplearn_save(fp, &learner);
        fclose(fp);
        return Py_BuildValue("i", 0);
    }
    return Py_BuildValue("i", -2);
}

static PyObject* load(PyObject* self, PyObject* args)
{
    FILE * fp;
    char * filename = NULL;
    int retval=-1;

    if (!PyArg_ParseTuple(args, "s", &filename))
        return Py_BuildValue("i", -1);

    fp = fopen(filename,"r");
    if (fp) {
        retval = deeplearn_load(fp, &learner, &random_seed);
        fclose(fp);
        if (retval == 0) {
            initialised = 1;
            return Py_BuildValue("i", 0);
        }
        return Py_BuildValue("i", -100 + retval);
    }
    return Py_BuildValue("i", -2);
}

static PyObject* plotHistory(PyObject* self, PyObject* args)
{
    int retval=-1;
    char * filename;
    char * title;
    int image_width=640, image_height=200;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "ssii", &filename, &title, &image_width, &image_height))
        return Py_BuildValue("i", -2);

    retval = deeplearn_plot_history(&learner,
                                    filename, title,
                                    image_width,
                                    image_height);

    return Py_BuildValue("i", retval);
}

/* performs a single training step */
static PyObject* training(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    return Py_BuildValue("i", deeplearndata_training(&learner));
}

/* returns the test set performance as a percentage */
static PyObject* getPerformance(PyObject* self, PyObject* args)
{
    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    return Py_BuildValue("f", deeplearndata_get_performance(&learner));
}

static PyObject* export(PyObject* self, PyObject* args)
{
    char * filename;

    if (initialised == 0) {
        return Py_BuildValue("i", -1);
    }

    if (!PyArg_ParseTuple(args, "s", &filename))
        return Py_BuildValue("i", -2);

    return Py_BuildValue("f", deeplearn_export(&learner, filename));
}

/*  define functions in module */
static PyMethodDef DeeplearnMethods[] =
{
    {"setSeed", setSeed, METH_VARARGS, "Sets the random seed"},
    {"init", init, METH_VARARGS, "Initialise a deep learner"},
    {"setLearningRate", setLearningRate, METH_VARARGS, "Sets the leraning rate in the range 0.0 - 1.0"},
    {"setDropoutsPercent", setDropoutsPercent, METH_VARARGS, "Sets the percentage of dropouts"},
    {"setErrorThreshold", setErrorThreshold, METH_VARARGS, "Sets the error threshold for training a layer"},
    {"setErrorThresholds", setErrorThresholds, METH_VARARGS, "Sets the training error thresholds from a list"},
    {"feedForward", feedForward, METH_VARARGS, "Perform network feed forward"},
    {"update", update, METH_VARARGS, "Update the network"},
    {"setInput", setInput, METH_VARARGS, "Sets the value of an input"},
    {"setInputs", setInputs, METH_VARARGS, "Sets the inputs from an array of floats"},
    {"setFields", setFields, METH_VARARGS, "Sets the input fields from an array which can contain numbers and strings"},
    {"setOutput", setOutput, METH_VARARGS, "Sets the desired value of an output"},
    {"getOutput", getOutput, METH_VARARGS, "Gets an output value"},
    {"getClass", getClass, METH_VARARGS, "Gets the output class"},
    {"setClass", setClass, METH_VARARGS, "Sets the output class"},
    {"setOutputs", setOutputs, METH_VARARGS, "Sets the desired outputs from an array of floats"},
    {"save", save, METH_VARARGS, "Save the network"},
    {"load", load, METH_VARARGS, "Load a network"},
    {"plotHistory", plotHistory, METH_VARARGS, "Plots the training history"},
    {"currentLayer", currentLayer, METH_VARARGS, "Returns the index of the hidden layer currently being pre-trained"},
    {"backpropError", backpropError, METH_VARARGS, "Returns backprop error for the network"},
    {"inputs", inputs, METH_VARARGS, "Returns the number of inputs"},
    {"outputs", outputs, METH_VARARGS, "Returns the number of outputs"},
    {"hiddens", hiddens, METH_VARARGS, "Returns the number of hidden units per layer"},
    {"layers", layers, METH_VARARGS, "Returns the number of hidden layers"},
    {"readCsvFile", readCsvFile, METH_VARARGS, "Reads the data from a csv file"},
    {"setHistoryPlotInterval", setHistoryPlotInterval, METH_VARARGS, "Sets the number of time steps after which to update the training history"},
    {"setPlotTitle", setPlotTitle, METH_VARARGS, "Sets the title of the training history graph"},
    {"training", training, METH_VARARGS, "Performs a training step"},
    {"getPerformance", getPerformance, METH_VARARGS, "Returns the test set performance as a percentage"},
    {"export", export, METH_VARARGS, "Exports the trained network as a standalone C program"},
    {"getErrorThreshold", getErrorThreshold, METH_VARARGS, "Returns an error threshold for the given layer"},
    {"test", test, METH_VARARGS, "Supply some inputs and returns the output values"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC

initdeeplearn(void) {
    (void) Py_InitModule("deeplearn", DeeplearnMethods);
}
