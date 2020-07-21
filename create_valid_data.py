import cv2
from numpy.lib.npyio import savetxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import os

#!!!NOTE THIS NEEDS TO RUN AFTER VALIDATION IMAGE FILES HAVE BEEN SELECTED/MOVED!!!
data_dir = op.join("C:\\", "Determine_Car_Speed", "data")
valid_imgs_dir = op.join(data_dir, "valid_data_imgs")

def validation_data():
    #*- validation data files range from 4200--8199
    offset = 4200
    set_size = len(os.listdir(valid_imgs_dir))
    val_data = np.zeros((set_size,66*220*3+1))
    for i in range(set_size):
        im1 = op.join(valid_imgs_dir, "valid_frame_" + str(i + offset) + ".jpg")
        im2 = op.join(valid_imgs_dir, "valid_frame_" + str(i + +offset + 1) + ".jpg")
        if (op.exists(im1) and op.exists(im2)):
            rgb_diff = opticalFlowDense(im1, im2)
            datapt = rgb_diff.ravel()
            datapt = np.append(datapt, training_labels[i + offset])
            val_data[i] = datapt
        else:
            break
    return val_data

val_data = validation_data()
np.save(op.join(data_dir, "validation_dataset.npy"), val_data)

#~~~~TO RETRIEVE DATA~~~~~
# def get_val_data(file):
#     init_arr = np.load(file)
#     init_arr = np.nan_to_num(init_arr)
#     val_labels = init_arr[:,-1]
#     val_labels = val_labels.reshape(-1,1)
#     val_data = init_arr[:,:-1]
#     val_data = val_data.reshape(-1,66,220,3)
#     return val_data, val_labels