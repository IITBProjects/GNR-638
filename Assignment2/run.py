import argparse
import os
import numpy as np
from codes import TwoLayerMLP
import json

def compare_gradients(num_samples,input_size,hidden_layer_size,output_size,numpy_seed,filepath):

    save_directory = os.path.dirname(filepath)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    np.random.seed(numpy_seed)
    inputs = np.random.rand(num_samples, input_size)
    inputs[:] = 1
    target = np.random.randint(0, 2, (num_samples, 1))
    model = TwoLayerMLP(input_size, hidden_layer_size, output_size)
    loss = model.forward(inputs,target)
    backprop_grads = model.backward()
    analytical_grads = model.analytical_gradients(inputs,target,epsilon=1e-6)
    overall_grads = {'backprop': {}, 'analytical': {},'difference': {}}

    print(f'Loss: {loss:.4f}')
    for key in backprop_grads.keys():
        diff = np.abs(backprop_grads[key] - analytical_grads[key])
        overall_grads['backprop'][key] = backprop_grads[key].tolist()
        overall_grads['analytical'][key] = analytical_grads[key].tolist()
        overall_grads['difference'][key] =  diff.tolist()

    # Save the differences to a file
    with open(filepath, 'w') as file:
        json.dump(overall_grads, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', action="store", dest="num_samples", default=1, type=int)
    parser.add_argument('--input_size', action="store", dest="input_size", default=2, type=int)
    parser.add_argument('--hidden_layer_size', action="store", dest="hidden_layer_size", default=2, type=int)
    parser.add_argument('--output_size', action="store", dest="output_size", default=1, type=int)
    parser.add_argument('--numpy_seed', action="store", dest="numpy_seed", default=42, type=int)
    parser.add_argument('--filepath', action="store", dest="filepath", default='data/gradients.json')

    args = parser.parse_args()

    compare_gradients(args.num_samples, args.input_size, args.hidden_layer_size, args.output_size, args.numpy_seed,args.filepath)