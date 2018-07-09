Sign language recognition system Using CNN

Demo Link:  https://www.youtube.com/watch?v=NsYyLi4dRSM

Languages, frameworks and tools used: Python, Keras, CNN, OpenCV, sckit-learn, Numpy, matplotlib, Floydhub

Program workflow: 
	1. python create_gestures.py: start to create the gestures by pressing 'c'
	2. python load_images.py: split the images to train, validation and test dataset
	3. python NormalCNN.py: The defined model is trained with 3 fold cross validation and stored with h5 extension
	4. python recognize_gesture_WithCV.py: This is the main application that predicts any gesture in real-time
	
You can display all gestures by running display_all_gestures.py
predict_batch.py and predict_OneImage are used for testing purposes


For more information, contact me: karimgreek@gmail.com