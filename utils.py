import sys
import importlib.util
from os.path import dirname, join, abspath
import numpy as np


def module_from_file(module_name, base_path, rel_path):
    file_path = join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.append(dirname(file_path))
    spec.loader.exec_module(module)
    return module


basepath = dirname(dirname(abspath(__file__)))
tfprocess = module_from_file("tfprocess", basepath, "lczero-training/tf/tfprocess.py")
leelaBoard = module_from_file("lcztools", basepath, "lczero_tools/src/lcztools/_leela_board.py")


def data_from_fen(fen):
    board = leelaBoard.LeelaBoard(fen)
    # board.push_uci('e2e4')
    print(board)
    return np.reshape(board.lcz_features(), [-1, 112, 8, 8])


def print_as_table(tensor, items):
    i = 0
    for num in tensor.flatten():
        print('{};{}'.format(i, num))
        i += 1
        if i >= items:
            break


def print_layers(model, fro, to, filter=None):
    for i in range(max(fro, 0), min(to, len(model.layers))):
        if filter is None or filter == 'all' or filter in model.layers[i].name:
            print('layer', i, model.layers[i].name, type(model.layers[i]), model.layers[i].output.shape)


def print_weights(model, filter=None):
    i = 0
    for weight in model.weights:
        if filter is None or filter == 'all' or filter in weight.name:
            print('weight', i, weight.name, weight.shape)
        i += 1


def softmax(array):
    return np.exp(array) / np.sum(np.exp(array))


def load_tensor_from_file(filename):
    print(filename)
    tensor = np.genfromtxt(filename,
                           delimiter=';',
                           comments='#',
                           usecols=(1,),
                           dtype=np.float32)
    shape = tensor.shape

    # anotated shape
    with open(filename, 'r') as file:
        shapeline = file.readline()
        print('shapeline', shapeline.strip()[1:])

        if shapeline[0] == '#':
            exec(shapeline[1:].strip())
            exec ('shape = (192, 1034)')
            print('shape', shape)

            exec('print(\'execing\')')
            tensor.shape = shape

    return tensor
