import numpy as np
import cv2 as cv
import os
from sklearn.cluster import KMeans

import pickle

# Imports from my project
import project_config



if __name__ == "__main__":
    filename     = f"{project_config.dir_root}/kmeans_model_test.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))


    new_data    = np.random.rand(3,128).astype('float32')
    pred_new = loaded_model.predict(new_data)

    x=1