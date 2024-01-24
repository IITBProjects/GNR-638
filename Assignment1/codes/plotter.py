import matplotlib.pyplot as plt
from .BoW import trainModel,testModel
from .BoW_with_glcm import trainModelWithGLCM,testModelWithGLCM
from sklearn.metrics import accuracy_score

def varyKAndPlot(train_path, test_path, max_clusters, kernel,task=1):
    k_values = range(10, max_clusters + 1, 10)  # You can adjust the range and step size as needed
    accuracies = []

    for k in k_values:
        print(f"Training and testing for K = {k}")
        if task ==1:
            kmeans, scale, svm, im_features = trainModel(train_path, k, kernel)
            true, predictions = testModel(test_path, kmeans, scale, svm, im_features, k, kernel, return_results=True)
        elif task==2:
            kmeans, scale, svm, im_features = trainModelWithGLCM(train_path, k, kernel)
            true, predictions = testModelWithGLCM(test_path, kmeans, scale, svm, im_features, k, kernel, return_results=True)
        accuracy = accuracy_score(true, predictions)
        print(f"Accuracy for K = {k} is {accuracy}")
        accuracies.append(accuracy)

    # Plotting K vs Accuracy
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Number of Clusters (K) vs Accuracy')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    return plt
