# Distracted Driver Detection
Classifying dashcam images to detect cases of distracted and safe driving.

# Environment
The library dependencies can be found in `requirements.txt`. All packages can be installed using `pip install <package_name>`. I have used Keras with the TensorFlow backend and Python 3 in my implementation. A special mention to Arun Ponnusamy for his repository `cvlib` which is available [here](https://github.com/arunponnusamy/cvlib).

# Files
The repository contains 4 files - `distracted_driver_detection.py`, which is the main file; `models.py`, which contains all the deep learning model definitions; `generators_and_training.py`, which contains all the data generators and the training functions and finally `data_utilities.py`, which contains several utility functions mostly dealing with file operations.

# Background and Data
Distracted drivers are a prime cause of road accidents in the world. In North America especially, using cellphones while driving is fairly commonplace, whether it be to change the music, texting, speaking on the phone or taking videos or photos. Identifying and preventing cases of distracted driving is an important task, the National Safety Council claims that 1 of every 4 car accidents in the United States is caused by a driver who is not paying attention to the road. Not too long ago State Farm, an insurance company, hosted a challenge on Kaggle to identify cases of distracted driving using computer vision. This is my implementation for that challenge. The Kaggle challenge can be found [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

The data is provided by State Farm itself and is available on Kaggle (follow the link above and head to the Data section). The dataset has around 16,000 training images and 79,000 testing images. The training images belong to 10 classes -
-  c0: Safe driving
-  c1: Texting (right)
-  c2: Talking (left)
-  c3: Texting (left)
-  c4: Talking (left)
-  c5: Operating the radio.
-  c6: Drinking / eating.
-  c7: Reaching behind.
-  c8: Hair and make-up.
- c9: Talking to the passenger.
The training images are split into ten directories each containing images belonging to one of the classes. Note that none of the people in the training images show up again in the testing images or vice versa. This makes the competition slightly more challenging. You can find more information about the dataset at the competition page.

My solution was used for a course project (without breaking any of the rules of the Kaggle competition). The focus is on the use of different techniques and seeing how different implementations compare to eachother.

# Implementation Details
I have used several techniques in the project. To give a gist, I have used a simple ConvNet architecture which is relatively shallow and the VGG16 architecture. I have also used an ensemble model and pseudo-labelling to generate new labeled data from the test images. You can find more information about pseudo-labeling [in this paper](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf) which proposed the technique and has a decent number of citations.
Before you run the program, there is a list of plugs that need to be set at the beginning of the main function in `distracted_driver_detection.py`. To correctly set these plugs some information about the implementation and the flow of the repository may be required. The basic flow is as follows:
1. Split the training data into three training and validation sets.
2. Split the training and validation sets into a _main_ set to use for the actual training and a _mini_ set to use while deciding the hyperparameters or babysitting the learning process.
3. Generate new data from the training and validation sets. This new data consists of hand crops and face crops of the drivers. The motivation behind this is that observing the hands would likely help detect some of the classes (think: operating the radio, reaching behind, hair and make-up, texting). On the other hand, observing the face would help detect some other classes (think: talking on the phone, talking to the passenger). These new datasets (hands and faces) also have their own main- and mini-sets. Faces are extracted using the `detect_faces()` function in `cvlib` and the hands are extracted by simply cropping a mid-portion of each image.
4. Run the single models (ConvNet and VGG16) individually on each of the three image crops (full image, hand crops, face crops). It is advisable to first run the models on the mini-sets of each crop and try to find the best values for the hyperparameters. One method is to start with a broad range of values for each hyperparameter and narrow down to a smaller range that works. Once you have your small range, training on the main-set will be much easier and time efficient.
5. At this stage there should be six model weight files - 3 for the ConvNet and 3 for VGG16 (one each for hand crops, face crops and full crops). Select the best weights file for each crops. These become the 3 loadpaths for the next stage of execution.
6. Next the ensemble model is trained using the best weight files from the previous stage. The ensemble model has three input arms - one arm trained on the full crops, one trained on the hands and one trained on the faces. The predictions from each arm are consolidated and presented as a single probability distribution. This consolidation is done by a neural network, so the ensemble model has to be trained as well. During training, the neural network learns the importance of the role of each arm in the final prediction.

![The ensemble architecture](images/ensemble.png)
7. [Optional] I trained the ensemble model using the mini-sets of each cropped dataset. My reasoning was that the ensemble should be trained using images that its components haven't seen before. For this reason, pseudo-labeling made sense to me (the mini-set has very few images).
8. [Optional] I used the ensemble (trained on mini-sets) to make predictions on a portion of the testing images. I then used these newly labeled test images to train my ensemble further and make predictions on the rest of the test dataset.

# Results and Notes
While training the single models (step 4), I trained ResNet50, Inceptionv3 and Xception in addition to the simple ConvNet and VGG16. These deeper models did not do very well however and I decided to stick with the two I mentioned. All models apart from the simple ConvNet were transfer learned from the Imagenet weight files available in Keras.

After training I found that VGG16 was much more confident and accurate than the simple ConvNet despite similar accuracy (see Plots 1 and 2). Finally, my ensemble showed better performance with all VGG16 arms - resulting in better pseudo-labels and consequently better final results. Note that you can use the main-sets to train the ensemble in step 7. Since the component models are frozen anyway, it will not make too much of a difference. In that case, there is no need for pseudo-labeling either.

![Loss vs Accuracy for the Simple ConvNet](images/simpleCNN-plot.png)
![Loss vs Accuracy for the VGG16](images/VGG16-plot.png)
