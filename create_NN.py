import numpy as np
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import os
import tensorflow as tf
import matplotlib.image as mpimg #?
import matplotlib.gridspec as gridspec #?
from sklearn.utils import shuffle #?
from sklearn.metrics import mean_squared_error #?
from sklearn.model_selection import train_test_split #?
from tqdm import tqdm #?
import h5py 
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard #?
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam, SGD

# ~~~~DEFINE DATA FILES~~~~~
data_dir = op.join("C:\\", "Determine_Car_Speed", "data")
training_labels = np.genfromtxt(op.join(data_dir, "train.txt"), delimiter="\n")
train_imgs_dir = op.join(data_dir, "train_data_imgs")
valid_imgs_dir = op.join(data_dir, "valid_data_imgs")
test_imgs_dir = op.join(data_dir, "test_data_imgs")
# flow_test_dir = op.join(data_dir, "flow_test_data")
# flow_train_dir = op.join(data_dir, "flow_train_data")

#~~~~HYPERPARAMETERS~~~~~
batch_size = 32 #16
num_epochs = 16 #25/49 #29/22
steps_per_epoch = 200 #38 #num_batches/num_epochs #400

#~~~~DATA PARAMETERS~~~~~
N_img_height = 66 #256 #66
N_img_width = 220 #640 #220
N_img_channels = 3

#~~~~DEFINE FUNCTIONS~~~~~
def add_rand_saturation(image, adj_factor):
    """
    Assign random adjustment to saturation value for each pixel in a frame.  
    Flow is BGR -> HLS -> transf. -> BGR
    """
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)  # -Convert to HLS
    # image_HLS[:,:,2] =np.full_like(image_HLS[:,:,2], np.random.uniform(0,255)) #-assign each frame a random saturation
    image_HLS[:, :, 2] = (
        image_HLS[:, :, 2] * adj_factor
    )  # -assign each frame a random saturation
    image_BGR = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)  ## Convert back to BGR

    return image_BGR

def opticalFlowDense(im1, im2):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    """
    # * Define random brightness adjustment factor
    rand_adj_factor = np.random.random() + 0.5
    # * Read in image 1, add random brightness, crop, resize
    image_current = cv2.imread(im1)
    image_current = add_rand_saturation(image_current, rand_adj_factor)
    image_current = image_current[100:356, :]
    image_current = cv2.resize(image_current, (220, 66), interpolation=cv2.INTER_AREA)
    # * Read in image 2, add random brightness, crop, resize
    image_next = cv2.imread(im2)
    image_next = add_rand_saturation(image_next, rand_adj_factor)
    image_next = image_next[100:356, :]
    image_next = cv2.resize(image_next, (220, 66), interpolation=cv2.INTER_AREA)
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    # hsv = np.zeros((im_H, im_W, 3))
    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    #     flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(
        gray_current,
        gray_next,
        flow_mat,
        image_scale,
        nb_images,
        win_size,
        nb_iterations,
        deg_expansion,
        STD,
        0,
    )

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow

def import_data_generator(data_dir, batch_size):
    # for some_file in sorted(os.listdir(data_dir), key=len):
    #     datafile = op.join(data_dir, some_file)
    #     current_data_file = np.load(op.join(data_dir,datafile))
    #     current_data_file = np.nan_to_num(current_data_file)
    #     current_labels = current_data_file[:,-1]
    #     current_labels = current_labels.reshape(-1,1)
    #     current_train_data = current_data_file[:,:-1]
    #     # current_train_data = current_train_data.reshape(-1,256,640,3)
    #     current_train_data = current_train_data.reshape(-1,66,220,3)
    #     yield current_train_data, current_labels
    # while True:
    #     data = np.random.rand(32, 256, 640, 3)
    #     label = np.random.rand(32,1)
    #     yield data, label
    while True:
        all_data_files = sorted(os.listdir(data_dir), key=len)
        lottery_nums = range(len(all_data_files)-1)
        data_pulled = np.zeros((batch_size,66,220,3))
        labels_pulled = np.zeros((batch_size,1))
        for i in range(batch_size):
            lucky_num = random.choice(lottery_nums)
            im1 = op.join(data_dir, "train_frame_" + str(lucky_num) + ".jpg")
            im2 = op.join(data_dir, "train_frame_" + str(lucky_num + 1) + ".jpg")
            while not (op.exists(im1) and op.exists(im2)):
                lucky_num = random.choice(lottery_nums)
                im1 = op.join(data_dir, "train_frame_" + str(lucky_num) + ".jpg")
                im2 = op.join(data_dir, "train_frame_" + str(lucky_num + 1) + ".jpg")
            rgb_diff = opticalFlowDense(im1, im2)
            data_pulled[i] = rgb_diff
            labels_pulled[i] = training_labels[lucky_num + 1]
        yield data_pulled, labels_pulled

def valid_data_generator(data_dir, batch_size):
    all_data_files = sorted(os.listdir(data_dir), key=len)
    lottery_nums = range(len(all_data_files)-1)
    data_pulled = np.zeros((batch_size,66,220,3))
    labels_pulled = np.zeros((batch_size,1))
    for i in range(batch_size):
        lucky_num = random.choice(lottery_nums)
        im1 = op.join(data_dir, "valid_frame_" + str(lucky_num) + ".jpg")
        im2 = op.join(data_dir, "valid_frame_" + str(lucky_num + 1) + ".jpg")
        while not (op.exists(im1) and op.exists(im2)):
            lucky_num = random.choice(lottery_nums)
            im1 = op.join(data_dir, "valid_frame_" + str(lucky_num) + ".jpg")
            im2 = op.join(data_dir, "valid_frame_" + str(lucky_num + 1) + ".jpg")
        rgb_diff = opticalFlowDense(im1, im2)
        data_pulled[i] = rgb_diff
        labels_pulled[i] = training_labels[lucky_num + 1]
    yield data_pulled, labels_pulled

def vehicle_speed():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/128.0 - 1.0, input_shape = inputShape))
    model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
    model.add(ELU())    
    model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
    model.add(ELU())    
    model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
    model.add(ELU())              
    model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))   
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    # sgd = SGD(lr=1e-05, momentum=0.999, clipnorm=1e0, nesterov=True)
    # model.compile(optimizer = sgd, loss = 'mean_absolute_error', metrics=['mean_squared_error'])
    # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-03)
    model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics=['mean_squared_error'])

    return model

model = vehicle_speed()

#~~~~TRAINING~~~~~
history = model.fit(import_data_generator(train_imgs_dir, batch_size),
                steps_per_epoch=steps_per_epoch, 
                epochs=num_epochs#,batch_size
                )

#~~~~SAVE MODEL~~~~~
model.save(op.join(data_dir, "vehicle_speed_adam.h5"))

#* Visualize history
#* Plot history: Loss
plt.plot(history.history['loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.savefig(op.join(data_dir, "Loss.png"))
#* Plot history: MSE
plt.plot(history.history['mean_squared_error'])
plt.title('Validation MSE history')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.savefig(op.join(data_dir, "MSE.png"))

#~~~~TEST/VALIDATE MODEL~~~~~
def get_val_data(file):
    init_arr = np.load(file)
    init_arr = np.nan_to_num(init_arr)
    val_labels = init_arr[:,-1]
    val_labels = val_labels.reshape(-1,1)
    val_data = init_arr[:,:-1]
    val_data = val_data.reshape(-1,66,220,3)
    return val_data, val_labels

# v_data, v_labels = get_val_data(op.join(data_dir, "validation_dataset.npy"))
# val_eval = model.evaluate(v_data, v_labels)
# print(val_eval)


#* TEST MODEL ON ALL TEST DATA
# all_test_files = sorted(os.listdir(test_data_dir), key=len)
# test_predictions = []
# test_predictions = np.array(test_predictions)
# for data_file in all_test_files:
#     print("Working on: ", data_file)
#     test_data = get_test_data(op.join(test_data_dir, data_file))
#     pred = model.predict(test_data)
#     test_predictions = np.append(test_predictions, pred)

# np.save(op.join(data_dir, "model_predictions.npy"), test_predictions)