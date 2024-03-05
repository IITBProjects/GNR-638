import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import os
import sys
from codes import CUB200,EfficientNetModel,MobileNetModel,InceptionNetModel,ResNetModel,train_model,save_lists_to_csv,plot_and_save_accuracy
from torchsummary import summary

# Dictionary mapping model names to their respective classes
MODEL_CLASSES = {
    'EfficientNetModel': EfficientNetModel,
    'MobileNetModel': MobileNetModel,
    'InceptionNetModel': InceptionNetModel,
    'ResNetModel': ResNetModel,
}

def train(torch_seed,dataset_path,resize_height,resize_width,model_name,batch_size,epochs,checkpoint_path,output_directory):
    if not os.path.exists(os.path.join(output_directory,model_name)):
        os.makedirs(os.path.join(output_directory,model_name))

    # Set the seed for the random number generators
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)

    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
    ])
    train_dataset = CUB200(dataset_path=dataset_path, data_type='train',transform=transform)
    test_dataset = CUB200(dataset_path=dataset_path, data_type='test',transform=transform)
    print("Train Dataset length:", train_dataset.__len__())
    print("Test Dataset length:", test_dataset.__len__())
    print("Dataset  Classes:", train_dataset.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the model based on the model name
    model_class = MODEL_CLASSES[model_name]
    model = model_class(num_classes=train_dataset.num_classes, freeze_layers=[])  # You may need to adjust other arguments

    # Optionally load model from checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint:", checkpoint_path)

    # Print model summary
    print("Model Summary:")
    summary(model, (3, resize_height, resize_width))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    training_loss_list,training_accuracy_list,test_loss_list,test_accuracy_list,epochs_list = train_model(train_loader, test_loader, model, criterion, optimizer, epochs)

    checkpoint_savepath = os.path.join(output_directory,model_name,'checkpoint.pth')
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs_list[-1]  # Save the last epoch
    }, checkpoint_savepath)
    print("Model checkpoint saved to:", checkpoint_savepath)
    save_lists_to_csv(training_loss_list, training_accuracy_list,test_loss_list, test_accuracy_list, epochs_list, os.path.join(output_directory,model_name))
    plot_and_save_accuracy(training_accuracy_list, test_accuracy_list, epochs_list,model_name, os.path.join(output_directory,model_name))
    os.system(f"mv logs.txt {os.path.join(output_directory,model_name,'logs.txt')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_seed', action="store", dest="torch_seed", default=1, type=int)
    parser.add_argument('--dataset_path', action="store", dest="dataset_path", default='dataset/CUB_200_2011')
    parser.add_argument('--resize_height', action="store", dest="resize_height", default=224, type=int)
    parser.add_argument('--resize_width', action="store", dest="resize_width", default=224, type=int)
    parser.add_argument('--model_name', action="store", dest="model_name", default='EfficientNetModel',choices=['EfficientNetModel','MobileNetModel','InceptionNetModel','ResNetModel'])
    parser.add_argument('--batch_size', action="store", dest="batch_size", default=16, type=int)
    parser.add_argument('--epochs', action="store", dest="epochs", default=5, type=int)
    parser.add_argument('--checkpoint_path', action="store", dest="checkpoint_path", default=None)
    parser.add_argument('--output_directory', action="store", dest="output_directory", default='outputs/')
    args = parser.parse_args()
    
    train(args.torch_seed,args.dataset_path,args.resize_height, args.resize_width,args.model_name, args.batch_size, args.epochs,args.checkpoint_path,args.output_directory)