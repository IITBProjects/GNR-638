import matplotlib.pyplot as plt
import pandas as pd
import os

def save_lists_to_csv(training_loss_list, training_accuracy_list, test_accuracy_list, epochs_list, output_directory):
    # Create a dataframe
    data = {
        'Epoch': epochs_list,
        'Training Loss': training_loss_list,
        'Training Accuracy': training_accuracy_list,
        'Test Accuracy': test_accuracy_list
    }
    df = pd.DataFrame(data)

    # Save dataframe to CSV
    df.to_csv(os.path.join(output_directory,'metrics.csv'), index=False)

def plot_and_save_accuracy(training_accuracy_list, test_accuracy_list, epochs_list, output_directory):
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, training_accuracy_list, label='Train Accuracy')
    plt.plot(epochs_list, test_accuracy_list, label='Test Accuracy')
    plt.title('Train and Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Save plot as image
    plt.savefig(os.path.join(output_directory,'accuracy_plot.png'))
    plt.close()