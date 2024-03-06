import os
import json
import torch
import pandas as pd
from codes import Pipeline, Utils

def main():
    config = json.load(open('config.json'))

    torch.manual_seed(config['torch_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['torch_seed'])

    output = os.path.join(config['output_dir'], config['train']['model'])
    os.makedirs(output, exist_ok = True)
            
    pipeline = Pipeline(config)
    pipeline.create_dataloder()
    pipeline.init_train()
    
    train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list = pipeline.train()
    pipeline.save()
    os.system(f"mv logs.txt {os.path.join(output, 'logs.txt')}")

    Utils.plot_metrics(train_loss_list, test_loss_list, "Loss", config['train']['model'], os.path.join(output, 'loss.png'))
    Utils.plot_metrics(train_accuracy_list, test_accuracy_list, "Accuracy", config['train']['model'], os.path.join(output, 'accuracy.png'))

    results = pd.DataFrame.from_dict({
        'Training Loss': train_loss_list,
        'Training Accuracy': train_accuracy_list,
        'Test Loss': test_loss_list,
        'Test Accuracy': test_accuracy_list
    })
    with open(os.path.join(output, 'metrics.txt'), 'w+') as f:
        f.write(results.__str__())

if __name__ == '__main__':
    main()
