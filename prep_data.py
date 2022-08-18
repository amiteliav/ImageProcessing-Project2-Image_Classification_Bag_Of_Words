import numpy as np
import cv2 as cv
import os
from sklearn.cluster import KMeans

# Imports from my project
import project_config

OMP_NUM_THREADS=2


def run_SIFT_example():
    """
    see SIFT in OpenCV: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
    """

    path = f"{project_config.dir_data_train}\Bedroom\image_0001.jpg"
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    # see: http://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    kp, des = sift.detectAndCompute(gray, None)
    # kp is a tuple with many details, des: is [n_des, 128]

    print(f"{des.shape=}")




# def run_kmeans_example():



def create_train_des():
    root, folders, _ = next(os.walk(project_config.dir_data_train))

    # Define a SIFT object to calc des'
    sift = cv.SIFT_create()

    # Create SIFT des folder for the train data
    if not os.path.exists(project_config.dir_sift_train):
        os.makedirs(project_config.dir_sift_train)

    # Run all over the dataset, and create a sift des' for each img
    for folder in sorted(folders):
        # Create SIFT des folder for the train data - create folder for each class
        des_class_folder = f"{project_config.dir_sift_train}/{folder}"
        if not os.path.exists(des_class_folder):
            os.makedirs(des_class_folder)

        folder_path = f"{root}/{folder}"
        _, _, files = next(os.walk(folder_path))
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):  # run only over images
                file_path = f"{folder_path}/{file}"
                img = cv.imread(file_path)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                _, des = sift.detectAndCompute(gray, None)  # clac img sift des'

                # save the des' for each image
                file_name = file.split(".")[0]  # if file is "XX.jpg" or "XX.jpeg" get only XX
                save_path = f"{project_config.dir_sift_train}/{folder}/{file_name}.npz"
                np.savez(save_path, des=des)


def create_test_des():
    root, folders, _ = next(os.walk(project_config.dir_data_test))

    # Define a SIFT object to calc des'
    sift = cv.SIFT_create()

    # Create SIFT des folder for the data
    if not os.path.exists(project_config.dir_sift_test):
        os.makedirs(project_config.dir_sift_test)

    # Run all over the dataset, and create a sift des' for each img
    for folder in sorted(folders):
        # Create SIFT des folder for the train data - create folder for each class
        des_class_folder = f"{project_config.dir_sift_test}/{folder}"
        if not os.path.exists(des_class_folder):
            os.makedirs(des_class_folder)

        folder_path = f"{root}/{folder}"
        _, _, files = next(os.walk(folder_path))
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):  # run only over images
                file_path = f"{folder_path}/{file}"
                img = cv.imread(file_path)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                _, des = sift.detectAndCompute(gray, None)  # clac img sift des'

                # save the des' for each image
                file_name = file.split(".")[0]  # if file is "XX.jpg" or "XX.jpeg" get only XX
                save_path = f"{project_config.dir_sift_test}/{folder}/{file_name}.npz"
                np.savez(save_path, des=des)


def load_all_des(mode="train"):
    if mode=="train":
        path = project_config.dir_sift_train
    elif mode=="test":
        path = project_config.dir_sift_test
    else:
        print("ERROR")

    root, folders, _ = next(os.walk(path))

    all_des = []
    for folder in sorted(folders):
        folder_path = f"{root}/{folder}"
        _, _, files = next(os.walk(folder_path))
        for file in files:
            load_path = f"{folder_path}/{file}"
            with np.load(load_path) as data:
                des = data['des']
                for i in range(len(des)):
                    all_des.append(des[i])

    all_des = np.array(all_des)
    # print(f"{all_des.shape=}")

    return all_des


def create_voc_bow(use_subset = 1000):
    all_des = load_all_des(mode="train")

    data2Kmeans = all_des
    if use_subset is not None:
        # Run over a subset of the data
        n_data     = all_des.shape[0]
        len_subset = int(min(n_data, n_data/use_subset))
        index = np.random.choice(n_data, len_subset, replace=False)
        data2Kmeans = all_des[index,:]
        print(f"{data2Kmeans.shape=}")

    # Define Kmeans object, and run it
    n_kmeans = project_config.n_kmeans
    kmeans   = KMeans(n_clusters=n_kmeans)

    # fit kmeans over the full/subset data
    kmeans.fit(data2Kmeans)

    # # Calc the kmeans prediction over the full dataset
    # all_des_pred = kmeans.predict(all_des)
    # print(f"{all_des_pred.shape=}")

    """ NOTE:
    Keamns object:
    param 'kmeans' holds alot of info about the clustering
        1. kmeans.cluster_centers_  : holds the centers, shape[n_kmeans, data len]
        2. kmeans.labels_           : holds the label in the range of [0,n_kmeans-1]
                                      for each of example in the data -> shape: (data len)
    """
    # ------ Save VOC ------
    # voc is [n_kmeans, 128], where 128 is the length of each des' from SIFT
    # later, using 'voc' we can create the bow for each img, and create it's histogram
    voc = kmeans.cluster_centers_
    np.savez(f"{project_config.dir_root}/voc.npz", voc=voc)
    # =================

    # Calc bow for each image in the train dataset
    bow = np.zeros((project_config.n_classes, project_config.n_kmeans, 100))  # 100 images for each class
    print(f"{bow.shape=}")

    root, folders, _ = next(os.walk(project_config.dir_sift_train))
    i = 0
    for folder in folders:
        j=0
        path_folder = f"{root}/{folder}"
        _, _, files = next(os.walk(path_folder))
        for file in files:
            file_path = f"{path_folder}/{file}"
            with np.load(file_path) as data:
                des = data['des']
            des_pred = kmeans.predict(des)

            # Create histogram for the img des'
            hist_bins = list(range(0,project_config.n_kmeans+1))
            hist = np.histogram(des_pred, bins=hist_bins)
            hist_sum = np.sum(hist[0])
            hist = hist[0]/hist_sum
            bow[i,:,j] = hist
            j+=1  # j is for files
            # -- end of all file
        i += 1  # i is for classes / folders
        #  -- end of all classes

    # save bow
    np.savez(f"{project_config.dir_root}/bow.npz", bow=bow)



def PrepareData():
    # ----- Set flags according to pre-calc or not -----
    # Sift des'
    flag_calc_train_des = False
    flag_calc_test_des  = False

    # create voc
    flag_calc_voc       = True
    use_subset          = None    # len of subset to fit the kmeans, or 'None' for full dataset,
    # =====================


    # ----- Calc' the SIFT des for train and test datasets ---
    if flag_calc_train_des is True:
        create_train_des()

    if flag_calc_test_des is True:
        create_test_des()
    # ------------------------------------------------

    # Create 'voc':
    if flag_calc_voc is True:
        create_voc_bow(use_subset=use_subset)


if __name__ == "__main__":
   # run_SIFT_example()

   PrepareData()
