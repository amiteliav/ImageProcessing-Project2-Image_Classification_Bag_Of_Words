import os



# Path to project directory
all_paths = []
dir_root       = f"."  # define the root folder for the project
dir_data       = f"{dir_root}/data"
dir_data_train = f"{dir_data}/train"
dir_data_test  = f"{dir_data}/test"
dir_sift_train = f"{dir_data}/sift_train"
dir_sift_test  = f"{dir_data}/sift_test"

dir_aux        = f"{dir_root}/aux"

# Path to projects voc, bow
path_voc = f"{dir_root}/voc.npz"
path_bow = f"{dir_root}/bow.npz"

# Parameters
sift_nfeatures = None # None or int number(like 200)  # in sift algorithm, return the best N keypoint. default=None
n_kmeans       = 50   # choose between 50-100
n_classes      = 15   # must be 15 for our dataset
n_KNN          = 9    # choose odd numbers, check between 3-15

dict_class_names = {
    0: 'Bedroom',
    1: 'Coast',
    2: 'Forest',
    3: 'Highway',
    4: 'Industrial',
    5: 'InsideCity',
    6: 'Kitchen',
    7: 'LivingRoom',
    8: 'Mountain',
    9: 'Office',
    10: 'OpenCountry',
    11: 'Store',
    12: 'Street',
    13: 'Suburb',
    14: 'TallBuilding',
}

