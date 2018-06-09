import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('tf')
from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#get the shapes of the image. We can read any image
def get_image_size():
	img = cv2.imread('/gestures/0/100.jpg', 0)
	return img.shape
# num of classes equal to the number of subfolders. In this project we have 46 classes
def get_num_of_classes():
	return len(os.listdir('/gestures/'))

image_x, image_y = get_image_size()


def cnn_model():
	num_of_classes = 27 # or use get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(64, (5,5), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.6))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	#metircs = ['accuracy'] means that the metric we want to optimize is accuracy
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="/output/karim_model_withCV.h5"
	# save the model as the name specified in filepath only when there is an improvement in the best accuracy on the validation dataset
	#checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	EarlyStop = EarlyStopping(monitor='val_acc', patience=20, mode='max')
	#callbacks_list stores the weights and the result of the saved model at the checkpoint
	callbacks_list = [EarlyStop]
	return model, callbacks_list

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cvscores = []
def train():
	with open("/divided_images/all_images", "rb") as f:
		all_images = np.array(pickle.load(f))
	with open("/divided_images/all_labels", "rb") as f:
		all_labels = np.array(pickle.load(f), dtype=np.int32)

	all_images = np.reshape(all_images, (all_images.shape[0], image_x, image_y, 1))
	model, callbacks_list = cnn_model()
	for train, test in kfold.split(all_images, all_labels):
		all_labels_cat = np_utils.to_categorical(all_labels)
		model.fit(all_images[train], all_labels_cat[train], epochs=50, batch_size=128)
		# we evaluate on test data which are test_images and test_labels
		scores = model.evaluate(all_images[test], all_labels_cat[test])
		#score[0] contains validation loss . score[1] contains validation accuracy
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	f = open( '/output/cvscores', 'w' )
	f.write(repr(cvscores))
	f.close()
	model.save("/output/karim_model_withCV.h5")


train()
K.clear_session();
