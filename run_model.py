#!/opt/homebrew/Caskroom/miniforge/base/bin/python
import argparse
import numpy as np
import yaml
import tensorflow as tf

from os.path import dirname, join, abspath
from utils import tfprocess, data_from_fen, print_as_table, softmax, print_layers, print_weights, load_tensor_from_file


allowed_heads = [
    'policy', 'policy_optimistic_st', 'policy_soft',
    'value_winner', 'value_q', 'value_q_err', 'value_st', 'value_st_err',
    'moves_left', 'attn_wts',
]
argparser = argparse.ArgumentParser(description='Run net or checkpoint in tensorflow.')
argparser.add_argument('--net',
                       type=str,
                       required=False,
                       help='Net file to be converted to a model and run.')
argparser.add_argument('--heads',
                       type=str,
                       required=False,
                       default='policy',
                       help=f"Output head to print. Comma-separated list of any of {allowed_heads}")
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
argparser.add_argument('--maxlen',
                       type=int,
                       required=False,
                       default=10000,
                       help='Maximum length of records to display. Used with "layer"')
argparser.add_argument('--dumpconfig',
                       type=bool,
                       required=False,
                       default=False,
                       help='Whether to dump the configuration file or not')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
if args.dumpconfig:
    print(yaml.dump(cfg, default_flow_style=False))

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()

if args.net:
    tfp.replace_weights(args.net)
else:
    tfp.restore()
# tfp.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
# input_var = tf.keras.Input(shape=(112, 8, 8))
# outputs = tfp.construct_net(input_var)
# tfp.model = tf.keras.Model(inputs=input_var, outputs=outputs)
# tfp.replace_weights(args.net)

# Run model
x = data_from_fen(args.fen)

# Get intermediate value
current_layer = args.layer

if (current_layer == -1 or current_layer is None) and args.show_layers is None \
        and args.layer is None and args.show_weights is None and args.weight is None:
    outputs = tfp.model(x, training=False)
    # print('layer', tfp.model.layers[0])
    # print('outputs', outputs)
    # tf.print(outputs, output_stream=sys.stderr)
    # print('outputs.data', outputs.data)
    # Heads
    for head in args.heads.split(','):
        if head not in allowed_heads:
            raise Exception(f"Cannot find specified head '{head}'. Allowed heads {allowed_heads}")
        output = outputs[head].numpy()
        if 'value' in head and 'err' not in head:
            output = softmax(output)
        print(f'{head} head')
        print_as_table(output, min(args.maxlen, output.flatten().shape[0]))
        print('\n')

    # print('policy', policy)
    # print('value', value)
    # print('moves_left', moves_left)

if args.show_layers:
    print_layers(tfp.model, 0, len(tfp.model.layers), filter=args.show_layers)

if current_layer != -1 and current_layer is not None:
    model = tf.keras.Model(tfp.model.input, tfp.model.layers[current_layer].output)
    outputs = model(x, training=False)
    print('layers', len(tfp.model.layers))
    print('current layer', current_layer, outputs.shape, tfp.model.layers[current_layer].name)
    print_layers(tfp.model, current_layer - 12, current_layer + 12)

    width = max(tfp.embedding_size, tfp.net.filters(), tfp.encoder_dff)  # , tfp.pol_encoder_dff)
    print_as_table(outputs.numpy(), min(args.maxlen, width * 8 * 8))

if args.weight is not None and not args.compare:
    weight = tfp.model.weights[args.weight]
    # print(weight.shape, weight.numpy().transpose())
    i = 0
    print('weights for {} {} {}'.format(args.weight, weight.shape, weight.name))
    for num in weight.numpy():
        print('{};{}'.format(i, num))
        i += 1
        # if i >= items:
        #     break
    # print_as_table(weight, min(args.maxlen, 1200))

if args.show_weights:
    print_weights(tfp.model, args.show_weights)


if (args.weight is not None or args.layer is not None) and args.compare:
    if args.weight is not None:
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
