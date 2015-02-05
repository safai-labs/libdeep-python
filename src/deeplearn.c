#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <Python.h>
#include <libdeep/globals.h>
#include <libdeep/deeplearn.h>

static int initialised = 0;
static deeplearn learner;
static unsigned int random_seed = 46362;
static float error_threshold[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

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
    error_threshold[layer_index] = threshold;
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
    
    if (!PyArg_ParseTuple(args, "s", &filename))
        return Py_BuildValue("i", -2);

    fp = fopen(filename,"r");
    if (fp) {
        deeplearn_load(fp, &learner, &random_seed);
        fclose(fp);
        return Py_BuildValue("i", 0);
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

/*  define functions in module */
static PyMethodDef DeeplearnMethods[] =
{
    {"setSeed", setSeed, METH_VARARGS, "Sets the random seed"},
    {"init", init, METH_VARARGS, "Initialise a deep learner"},
    {"setLearningRate", setLearningRate, METH_VARARGS, "Sets the leraning rate in the range 0.0 - 1.0"},
    {"setDropoutsPercent", setDropoutsPercent, METH_VARARGS, "Sets the percentage of dropouts"},
    {"setErrorThreshold", setErrorThreshold, METH_VARARGS, "Sets the error threshold for training a layer"},
    {"feedForward", feedForward, METH_VARARGS, "Perform network feed forward"},
    {"update", update, METH_VARARGS, "Update the network"},
    {"setInput", setInput, METH_VARARGS, "Sets the value of an input"},
    {"setOutput", setOutput, METH_VARARGS, "Sets the desired value of an output"},
    {"save", save, METH_VARARGS, "Save the network"},
    {"load", load, METH_VARARGS, "Load a network"},
    {"plotHistory", plotHistory, METH_VARARGS, "Plots the training history"},
    {"inputs", inputs, METH_VARARGS, "Returns the number of inputs"},
    {"outputs", outputs, METH_VARARGS, "Returns the number of outputs"},
    {"hiddens", hiddens, METH_VARARGS, "Returns the number of hidden units per layer"},
    {"layers", layers, METH_VARARGS, "Returns the number of hidden layers"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC

initdeeplearn(void) {
    (void) Py_InitModule("deeplearn", DeeplearnMethods);
}
