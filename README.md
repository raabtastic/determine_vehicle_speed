# determine_vehicle_speed
Determine vehicle speed from dashcam video. Note this implementation requires TensorFlow, Keras, and OpenCV.

MSE on validation set (~20% of the training set) is 1.49521.

Files included:
* create_test_data.py
   * Creates test images from a test video
* create_training_data.py
   * Creates training images from a training video
* create_valid_data.py
   * Creates validation data from sequestered training images
   * (Note that 4000 contiguous training images were reserved for validation.)
* create_NN.py
   * Creates a CNN and trains on training data
   * Training data is given as the (rescaled and regularized) dense (Farneback) optical flow between adjacent frames.
* vehicle_speed_adam.h5
   * Trained model saved in h5 format.
* model_predications.npy
   * Model predictions for test.mp4 
   * Each value is a speed prediction for a frame, starting on the second frame
