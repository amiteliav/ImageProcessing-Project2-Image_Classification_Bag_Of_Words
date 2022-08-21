import numpy as np
import cv2 as cv
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


import pickle

# Imports from my project
import project_config
from prep_data import des2hist


def pred_label_from_des_pred(des_pred):

    # Turn the predicted kmeans to histogram - img bow
    hist = des2hist(des_pred)


    # == Run KNN to get the best label for the given img ===
    """
    To use the KNN we should:
        1. fit the training data:
            1.1 - load bow.npz from training (shape of [n_classes, VOC len, 100] )
            1.2 reshape it [n_classes*100 , VOC len]
            1.3 create labeling vector to fit the KNN model, shape: [n_classes*100,]
            1.4 fit the KNN model
        2. predict label given a new test image bow
    """
    with np.load(project_config.path_bow) as data:
        bow = data['bow']

    n_classes, voc_len, n_samples = bow.shape
    bow_to_knn = np.swapaxes(bow, 1, 2)
    bow_to_knn = bow_to_knn.reshape(n_classes*n_samples, voc_len)

    labels_to_knn = np.array(())
    for i in range(n_classes):
        labels = np.repeat(i, n_samples)
        labels_to_knn = np.concatenate((labels_to_knn,labels), axis=0)

    knn = KNeighborsClassifier(n_neighbors=project_config.n_KNN)
    knn.fit(bow_to_knn, labels_to_knn)

    label_pred_int = int(knn.predict(hist.reshape(1, -1)))
    # ======================

    return label_pred_int


def convert_class_labels(label_results):
    """
    Convert the int-labels to the classes str-labels
    :param label_results: int list
    :return: string list
    """
    results = []
    for int_label in label_results:
        str_label = project_config.dict_class_names[int_label]
        results.append(str_label)

    return results


def ClassifyImg(PathToImg, create_txt=True):
    """
    input: PathToImg: a folder path holding the test images, no subfolders

    --------------
    for each images does the follow:
        1. get the SIFT des'
        2. use the trained Kmeans model and retrieve the img labeling
        3. turn the pred_labels from Kmeans to hist -> create img BOW
        4. use the trainign data and KNN to classify the img
        5. convert int labels to string class-names

    NOTES!:
        1. I assume prep_data.py and PrepareData() already run
            which created the subfolders, kmeans model.
    """

    _, _, files = next(os.walk(PathToImg))


    # --- Create a subfolder for the des' -----
    class_test_des_path = f"{PathToImg}/des"
    if not os.path.exists(class_test_des_path):
        os.makedirs(class_test_des_path)
    # ------------------------------------------


    # --- SIFT -------
    sift = cv.SIFT_create()
    for file in sorted(files):
        if file.endswith(".jpg") or file.endswith(".jpeg"):  # run only over images
            file_path = f"{PathToImg}/{file}"
            img  = cv.imread(file_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, des = sift.detectAndCompute(gray, None)  # calc img sift des'

            # save the des' for each image
            file_name = file.split(".")[0]  # if file is "XX.jpg" or "XX.jpeg" get only XX
            save_path = f"{class_test_des_path}/{file_name}.npz"
            np.savez(save_path, des=des)
    print(f"Finish calc all SIFT des for the classifier test")
    # --------------------------


    # Load the Kmeans model and predict the data
    path_to_model = f"{project_config.dir_root}/kmeans_model.pkl"
    kmeans = pickle.load(open(path_to_model, 'rb'))
    print(f"Finish loading kmeans model")


    # -- Load the bow file, and create KNN model ---
    """
       To use the KNN we should:
           1. fit the training data:
               1.1 - load bow.npz from training (shape of [n_classes, VOC len, 100] )
               1.2 reshape it [n_classes*100 , VOC len]
               1.3 create labeling vector to fit the KNN model, shape: [n_classes*100,]
               1.4 fit the KNN model
           2. predict label given a new test image bow
       """
    with np.load(project_config.path_bow) as data:
        bow = data['bow']

    n_classes, voc_len, n_samples = bow.shape
    bow_to_knn = np.swapaxes(bow, 1, 2)
    bow_to_knn = bow_to_knn.reshape(n_classes * n_samples, voc_len)

    labels_to_knn = np.array(())
    for i in range(n_classes):
        labels = np.repeat(i, n_samples)
        labels_to_knn = np.concatenate((labels_to_knn, labels), axis=0)

    knn = KNeighborsClassifier(n_neighbors=project_config.n_KNN)
    knn.fit(bow_to_knn, labels_to_knn)
    # -----------------------------------


    label_results = []

    # predict with kmeans over the class-test dataset
    _, _, files = next(os.walk(class_test_des_path))
    for file in files:
        file_path = f"{class_test_des_path}/{file}"
        with np.load(file_path) as data:
            des = data['des']
        des_pred = kmeans.predict(des)
        hist = des2hist(des_pred)          # Turn the predicted kmeans to histogram - img bow
        label = int(knn.predict(hist.reshape(1, -1)))
        label_results.append(label)
    print(f"Finish predicting labels")


    # Convert int-labels to str-labels
    if create_txt is True:
        results = convert_class_labels(label_results)
        with open(f"{project_config.dir_root}/results.txt", "w") as file:
            for item in results:
                file.write("%s\n" % item)
            file.close()
        print(f"results.txt is ready")


    return label_results

def classifier_test_all():
    """
    this code test my model over the whole test dataset
    :return: prints the accuracy
    """
    tot_samples = 0
    tot_correct = 0
    for cls in range(project_config.n_classes):
        str_label            = project_config.dict_class_names[cls]
        path_classifier_test = f"{project_config.dir_data_test}/{str_label}"

        PathToImg = path_classifier_test
        results   = ClassifyImg(PathToImg, create_txt = False)

        cls_n_samples = len(results)
        cls_correct   = results.count(cls)

        tot_samples+=cls_n_samples
        tot_correct+=cls_correct

        # Print per-class results
        print(f"----------- Results class: {cls} ---------")
        print(f"results: {cls_correct}/{cls_n_samples}")
        print(f"acc (between 0-1): {(cls_correct / cls_n_samples):.4f}")
        print("-----------------------------")
    # ---------

    print("======================================================")

    # total result over all the classes
    print(f"----------- Final results over all the classes ---------")
    print(f"results: {tot_correct}/{tot_samples}")
    print(f"acc (between 0-1): {(tot_correct / tot_samples):.4f}")
    print("-----------------------------")




if __name__ == "__main__":
    """
    :param 'path_classifier_test': path to the test dataset folder
        
    """
    # #  --- Classifier for submission ---
    # path_classifier_test = f"{project_config.dir_data}/classifier_test"
    #
    # PathToImg = path_classifier_test
    # _ = ClassifyImg(PathToImg)
    # # ----------------------------------


    # --- Testing my model over all the Test dataset ---
    classifier_test_all()
    # -------------------------

