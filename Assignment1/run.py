import argparse
import os
from codes import varyKAndPlot

def task1(train_path, test_path, max_clusters, kernel_type,plot_path):
    if not (kernel_type == "linear" or kernel_type == "precomputed"):
        print("Kernel type must be either linear or precomputed")
        return

    plt = varyKAndPlot(train_path, test_path, int(max_clusters), kernel_type,task=1)
     # Check if the directory for saving the plot exists, create it if not
    plot_directory = os.path.dirname(plot_path)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(plot_path)  

def task2(train_path, test_path, max_clusters, kernel_type,plot_path):
    if not (kernel_type == "linear" or kernel_type == "precomputed"):
        print("Kernel type must be either linear or precomputed")
        return

    plt = varyKAndPlot(train_path, test_path, int(max_clusters), kernel_type,task=2)
     # Check if the directory for saving the plot exists, create it if not
    plot_directory = os.path.dirname(plot_path)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(plot_path)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', action="store", dest="task", required=True, choices=['1', '2'])
    args, remaining_args = parser.parse_known_args()

    if args.task == '1':
        task1_args_parser = argparse.ArgumentParser()
        task1_args_parser.add_argument('--train_path', action="store", dest="train_path", default='dataset/train')
        task1_args_parser.add_argument('--test_path', action="store", dest="test_path", default='dataset/test')
        task1_args_parser.add_argument('--max_clusters', action="store", dest="max_clusters", default=100)
        task1_args_parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")
        task1_args_parser.add_argument('--plot_path', action="store", dest="plot_path", default="images/task1.png")
        task1_args = task1_args_parser.parse_args(remaining_args)
        task1(task1_args.train_path, task1_args.test_path, task1_args.max_clusters, task1_args.kernel_type,task1_args.plot_path)

    elif args.task == '2':
        task2_args_parser = argparse.ArgumentParser()
        task2_args_parser.add_argument('--train_path', action="store", dest="train_path", default='dataset/train')
        task2_args_parser.add_argument('--test_path', action="store", dest="test_path", default='dataset/test')
        task2_args_parser.add_argument('--max_clusters', action="store", dest="max_clusters", default=100)
        task2_args_parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")
        task2_args_parser.add_argument('--plot_path', action="store", dest="plot_path", default="images/task2.png")
        task2_args = task2_args_parser.parse_args(remaining_args)
        task2(task2_args.train_path, task2_args.test_path, task2_args.max_clusters, task2_args.kernel_type,task2_args.plot_path)

    
