import cv2
from numpy.lib.npyio import savetxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import os

# ~~~~DEFINE IMAGE AND VIDEO FILES~~~~~
data_dir = op.join("C:\\", "Determine_Car_Speed", "data")
train_vid = op.join(data_dir, "train.mp4")
train_imgs_dir = op.join(data_dir, "train_data_imgs")
flow_train_dir = op.join(data_dir, "flow_train_data")
training_labels = np.genfromtxt(op.join(data_dir, "train.txt"), delimiter="\n")
im_H = 66 #256
im_W = 220 #640

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

    hsv = np.zeros((im_H, im_W, 3))
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

#! Process training data in two steps. 1) create image files and 2) compute/save flows
#! Could process in one step but this creates unnecessary fragility

#~~~~CREATE IMAGE FILES FROM TRAIN VIDEO~~~~~
#* The video feed is read in as a VideoCapture object
vidcap = cv2.VideoCapture(train_vid)
success, image = vidcap.read()
frames_count=0
while success:
    cv2.imwrite(op.join(train_imgs_dir, "train_frame_{:d}.jpg".format(frames_count)), image) #* save frame as image file
    print("Created file: ", op.join(train_imgs_dir, "train_frame_{:d}.jpg".format(frames_count)))
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    frames_count += 1

#~~~~CREATE OPTICAL FLOW DATASET~~~~~
#? IGNORE BECAUSE OPTICAL FLOW W/ RAND. SAT. IS DONE ON-THE-FLY
# all_train_imgs = os.listdir(train_imgs_dir)
# ex_length = im_H * im_W * 3 + 1  #_ flattened rgb_diff output plus 1 for label
# batch_size = 16 #_ Create data files with 16 or 32 training examples in each. Use .npy files for speed
# train_data = np.zeros((batch_size, ex_length))
# batch_num = 0
# batch_iter = 0
# ex_to_do = len(all_train_imgs)

# while ex_to_do > 0:
#     print("batch_num = {:n}\nbatch_iter = {:n}\n".format(batch_num, batch_iter))
#     print("{:n} examples left to do".format(ex_to_do))
#     i = len(all_train_imgs) - ex_to_do

#     im1 = op.join(train_imgs_dir, "train_frame_" + str(i) + ".jpg")
#     im2 = op.join(train_imgs_dir, "train_frame_" + str(i + 1) + ".jpg")

#     #* If both files exist, compute flow
#     if op.exists(im1) and op.exists(im2):
#         #* Compute optical flow, unravel, append target label, store in array
#         rgb_diff = opticalFlowDense(im1, im2)
#         rgb_diff = rgb_diff.ravel() #_ on load, reshape to (66, 220, 3)
#         rgb_diff = np.append(rgb_diff, training_labels[i + 1])
#         train_data[batch_iter] = rgb_diff
#     else:
#         print("ran out of files\nsaving and breaking")
#         mask = [i for i in range(len(train_data)) if sum(train_data[i]) != 0.0] #_ create mask to filter out residual zeros
#         train_data = train_data[mask]#_ take non-zero elements of train_data
#         np.save(
#             op.join(flow_train_dir, "flow_train_"+str(batch_num)+".npy"), train_data
#         )
#         break

#     #* If end of batch, save data
#     if batch_iter == batch_size - 1:
#         #* Save current train_data array to file
#         np.save(
#             op.join(flow_train_dir, "flow_train_"+str(batch_num)+".npy"), train_data
#         )
#         #* Reset batch_iter, update batch_num
#         batch_iter = 0
#         batch_num += 1
#         #* Reset train_data array to zeros
#         train_data = np.zeros((batch_size, ex_length))
#     elif batch_iter < (batch_size - 1):
#         batch_iter += 1
    
#     #* Update num to do
#     ex_to_do -= 1