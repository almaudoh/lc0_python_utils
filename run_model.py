import argparse
import sys
import numpy as np
import yaml
import tensorflow as tf

import importlib.util
from os.path import dirname, join, abspath


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

argparser = argparse.ArgumentParser(description='Run net or checkpoint in tensorflow.')
argparser.add_argument('net',
                       type=str,
                       help='Net file to be converted to a model checkpoint.')
argparser.add_argument('--cfg',
                       type=argparse.FileType('r'),
                       required=True,
                       help='yaml configuration with training parameters')
argparser.add_argument('--fen',
                       type=str,
                       default='startpos',
                       help='FEN position to run NN evaluation. Default is startpos.')
argparser.add_argument('--show-weights',
                       type=str,
                       default=None,
                       help='Show all the weights in the model (if "all"), or weights whose name match the string.')
argparser.add_argument('--show-layers',
                       type=str,
                       default=None,
                       help='Show all the layers in the model (if "all"), or layers whose name match the string.')
argparser.add_argument('--layer',
                       type=int,
                       help='Show the activations of the specified layer.')
argparser.add_argument('--weight',
                       type=int,
                       help='Show info on the specified weight.')
argparser.add_argument('--compare',
                       type=argparse.FileType('r'),
                       help='Compare the weight or layer tensor with tensor in this file.')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()
# tfp.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
# input_var = tf.keras.Input(shape=(112, 8, 8))
# outputs = tfp.construct_net(input_var)
# tfp.model = tf.keras.Model(inputs=input_var, outputs=outputs)
tfp.replace_weights(args.net)


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


def print_layers(fro, to, filter=None):
    for i in range(max(fro, 0), min(to, len(tfp.model.layers))):
        if filter is None or filter == 'all' or filter in tfp.model.layers[i].name:
            print('layer', i, tfp.model.layers[i].name, type(tfp.model.layers[i]), tfp.model.layers[i].output.shape)


def print_weights(filter=None):
    i = 0
    for weight in tfp.model.weights:
        if filter is None or filter == 'all' or filter in weight.name:
            print('weight', i, weight.name, weight.shape)
        i += 1


def softmax(array):
    return np.exp(array) / np.sum(np.exp(array))


# Run model
x = data_from_fen(args.fen)

# Get intermediate value
current_layer = args.layer

if (current_layer == -1 or current_layer is None) and not args.show_layers \
        and not args.layer and not args.show_weights and not args.weight:
    outputs = tfp.model(x, training=False)
    # print('layer', tfp.model.layers[0])
    # print('outputs', outputs)
    # tf.print(outputs, output_stream=sys.stderr)
    # print('outputs.data', outputs.data)
    # Policy
    print_as_table(outputs[0].numpy(), 1858)
    print_as_table(softmax(outputs[1].numpy()), 3)
    if len(outputs) > 2:
        print_as_table(outputs[2].numpy(), 1)
    # print('policy', policy)
    # print('value', value)
    # print('moves_left', moves_left)

if args.show_layers:
    print_layers(0, len(tfp.model.layers), filter=args.show_layers)

if current_layer != -1 and current_layer is not None:
    model = tf.keras.Model(tfp.model.input, tfp.model.layers[current_layer].output)
    outputs = model(x, training=False)
    print('layers', len(tfp.model.layers))
    print('current layer', current_layer, outputs.shape, tfp.model.layers[current_layer].name)
    print_layers(current_layer - 12, current_layer + 12)

    # print_as_table(outputs[0].numpy(), 8 * 8)
    width = max(tfp.embedding_size, tfp.net.filters(), tfp.embedding_size, tfp.encoder_dff, tfp.pol_encoder_dff)
    print_as_table(outputs.numpy(), width * 8 * 8)

if args.weight and not args.compare:
    weight = tfp.model.weights[args.weight]
    # print(weight.shape, weight.numpy().transpose())
    i = 0
    print('weights for {} {} {}'.format(args.weight, weight.shape, weight.name))
    for num in weight.numpy():
        print('{};{}'.format(i, num))
        i += 1
        # if i >= items:
        #     break
    # print_as_table(weight, 1200)

if args.show_weights:
    print_weights(args.show_weights)


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


if (args.weight or args.layer) and args.compare:
    if args.weight:
        # Leela's weights are transposed in the model.
        tensor = np.transpose(tfp.model.weights[args.weight])
    else:
        model = tf.keras.Model(tfp.model.input, tfp.model.layers[args.layer].output)
        outputs = model(x, training=False)
        tensor = outputs.numpy()

    compare = load_tensor_from_file(join(dirname(abspath(__file__)), args.compare.name))

    # print(tensor)
    # print(compare)
    if compare.shape != tensor.shape:
        print('Different shapes: ', tensor.shape, compare.shape)

    else:
        diff = tensor - compare
        mse = np.mean(np.square(diff))
        # mse = np.square(tensor - compare)
        print('MSE: ', mse,
              'Max: ', np.max(diff),
              'Min: ', np.min(diff),
              'Mean: ', np.mean(diff),
              'Stddev: ', np.std(diff))
        print(diff)
