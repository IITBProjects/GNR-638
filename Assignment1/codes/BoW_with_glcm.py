import cv2
import numpy as np 
from sklearn.preprocessing import StandardScaler
from .helper import getFiles,readImage,getDescriptors,vstackDescriptors,clusterDescriptors,extractFeatures,plotHistogram,findSVM,plotConfusions,findAccuracy
from .helper import extract_glcm_features,concat_kmeans_and_glcm

def trainModelWithGLCM(path, no_clusters, kernel, debug=False):
    images = getFiles(True, path)
    print("Train images path detected.")
    sift = cv2.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 7
    image_count = len(images)

    for img_path in images:
        if("city" in img_path):
            class_index = 0
        elif("face" in img_path):
            class_index = 1
        elif("green" in img_path):
            class_index = 2
        elif("house_building" in img_path):
            class_index = 3
        elif("house_indoor" in img_path):
            class_index = 4
        elif("office" in img_path):
          class_index = 5
        else:
          class_index = 6


        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    glcm_features = extract_glcm_features(images)
    print("GLCM features extracted.")

    features_combined = concat_kmeans_and_glcm(im_features, glcm_features)

    # Explicitly perform l2-norm on the features
    features_combined = features_combined / np.linalg.norm(features_combined, axis=1, keepdims=True)

    scale = StandardScaler().fit(features_combined)
    features_combined = scale.transform(features_combined)
    print("Train images normalized.")

    if debug:
        plotHistogram(features_combined, no_clusters + len(glcm_features[0]))
        print("Features histogram plotted.")

    svm = findSVM(features_combined, train_labels, kernel)
    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm, features_combined


def testModelWithGLCM(path, kmeans, scale, svm, im_features, no_clusters, kernel, return_results=False, debug=False):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "city",
        "1": "face",
        "2": "green",
        "3": "house_building",
        "4": "house_indoor",
        "5": "office",
        "6": "sea"
    }

    sift = cv2.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        if(des is not None):
            count += 1
            descriptor_list.append(des)

            if("city" in img_path):
                true.append("city")
            elif("face" in img_path):
                true.append("face")
            elif("green" in img_path):
                true.append("green")
            elif("house_building" in img_path):
                true.append("house_building")
            elif("house_indoor" in img_path):
                true.append("house_indoor")
            elif("office" in img_path):
                true.append("office")
            else:
                true.append("sea")
       
    descriptors = vstackDescriptors(descriptor_list)
    test_im_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    test_glcm_features = extract_glcm_features(test_images)

    test_features_combined = concat_kmeans_and_glcm(test_im_features, test_glcm_features)

    # Explicitly perform l2-norm on the features
    test_features_combined = test_features_combined / np.linalg.norm(test_features_combined, axis=1, keepdims=True)

    test_features_combined = scale.transform(test_features_combined)

    kernel_test = test_features_combined
    if kernel == "precomputed":
        kernel_test = np.dot(test_features_combined, im_features.T)

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    if debug:
        plotConfusions(true, predictions)
        print("Confusion matrixes plotted.")

        findAccuracy(true, predictions)
        print("Accuracy calculated.")
        print("Execution done.")

    if return_results:
        return true, predictions
    