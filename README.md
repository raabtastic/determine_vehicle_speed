# determine_vehicle_speed
Determine vehicle speed from dashcam video. Note this implementation requires TensorFlow, Keras, and OpenCV.

Files included:
* create_test_data.py
   * Creates test images from a test video
* create_training_data.py
   * Creates training images from a training video
   * (Note that some training data should be reserved for validation. Python code not provided here for that.)
* Create_NN.py
   * Creates a CNN and trains on training data
   * Training data is given as the (rescaled and regularized) dense (Farneback) optical flow between adjacent frames.
* model.h5
   * Model information for trained model.
* Plots:
