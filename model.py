import numpy as np
import os
import cv2
import csv
import math
from pathlib import Path
import pandas as pd
import random

import tensorflow as tf


from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential

from keras import backend as K
from keras.layers.core import Flatten, Dense, Lambda
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, ELU
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import json


#new sizes to fit NVIDIA model
col, row = 200,66
    


#Loading driving_log cvs file, using pandas framework
csv_file = 'driving_log.csv'
data_file = pd.read_csv(csv_file,
                         index_col = False)
data_file.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

def crop_image(image):

    shape = image.shape
    
    #Cut off the sky from the original picture, so that we train our network only in relevant data.
    crop_up = int(shape[0]/5)
    
    #Cut off the front of the car
    crop_down = shape[0]-25

    image = image[crop_up:crop_down, 0:shape[1]]
    
    #Resize to fit data for NVIDIA model
    image = cv2.resize(image,(col,row), interpolation=cv2.INTER_AREA) 
    
    return image


def preprocess_image_training(row_data, noise_factor=0.04):
    #Following the advise of Annie Flippo, to introduce some noise on the angles
    noise = (random.random() - 0.5) * 2.0 * 1.2 * noise_factor
    angle = row_data['steering'] * noise

    orientation = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    # to simulate recovery
    if orientation == 'left':
        path_file = row_data['left'][0].strip()
        angle += 0.25
    elif orientation == 'right':
        path_file = row_data['right'][0].strip()
        angle -= 0.25
        
    else:
        path_file = row_data['center'][0].strip()
        angle -= 0

    #Preprocessing images
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = crop_image(image)
    image = np.array(image)
    
    #Flipping randomly half of the images for aumentagtion
    #to compensate for data biased to left turns
    prob_flip = np.random.randint(2)
    if prob_flip==0:
        image = cv2.flip(image,1)
        angle = -angle
    
    return image, angle

    
    
def preprocess_image_prediction(row_data):
    path_file = row_data['center'][0].strip()
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = crop_image(image)
    image = np.array(image)
    return image
    
    
def generator_train(data, batch_size = 32):
    batch_images = np.zeros((batch_size, row, col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for batch in range(batch_size):
            line = np.random.randint(len(data))
            line_data = data.iloc[[line]].reset_index()
            image,angle = preprocess_image_training(line_data)
            
            batch_images[batch] = image
            batch_steering[batch] = angle
        
        yield batch_images, batch_steering


def generator_validation(data):
    # Validation generator
    while 1:
        for line in range(len(data)):
            row = data.iloc[[line]].reset_index()
            image = preprocess_image_prediction(row)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            angle = row['steering'][0]
            angle = np.array([[angle]])
            yield image, angle
            
            
            
def save_model(file_JSON,file_Weights, model_obj):
    #Check if an old version of the file exists remove it is if True
    if Path(file_JSON).is_file():
        os.remove(file_JSON)
    json_string = model_obj.to_json()
    
    with open(file_JSON,'w' ) as f:
        json.dump(json_string, f)
    
    #Check if an old version of the file exists remove it if True
    if Path(file_Weights).is_file():
        os.remove(file_Weights)
    model_obj.save_weights(file_Weights)         
    
    

def get_model():
    #Using NVIDIA model.
    model = Sequential()
    #Appliying lambda for normalization
    model.add(Lambda(lambda x:x/127.5 -1., input_shape = (row,col,3)))
    model.add(Convolution2D(24, 5, 5,border_mode='valid',subsample=(2,2),     init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, border_mode='valid',subsample=(2,2), init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, border_mode='valid',subsample=(2,2),init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1),init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1),init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164,init='he_normal'))
    model.add(ELU())
    model.add(Dense(100,init='he_normal'))
    model.add(ELU())
    model.add(Dense(50,init='he_normal'))
    model.add(ELU())
    model.add(Dense(10,init='he_normal'))
    model.add(ELU())
    model.add(Dense(1,name='output',init='he_normal'))
    
    
    learning_rate = 0.0001
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


model = get_model()

val_generator = generator_validation(data_file)

val_size = len(data_file)
batch_size = 256

best_model = 0
best_validation = 1000
validations_score = []


for i in range(10):

    generator = generator_train(data_file,batch_size)

    history = model.fit_generator(generator,
            samples_per_epoch=20224, nb_epoch=1, validation_data= val_generator, nb_val_samples= val_size)
    
    file_JSON = 'model_' + str(i) + '.json'
    file_Weights = 'model_' + str(i) + '.h5'
    
    save_model(file_JSON, file_Weights, model)
    
    val_loss = history.history['val_loss'][0]
    if val_loss < best_validation:
        best_model = i
        best_validation = val_loss
        file_JSON = 'model_best.json'
        file_Weights = 'model_best.h5'
        save_model(file_JSON,file_Weights,model)

    validations_score.append([i, val_loss])


with open('log_val.csv', 'w', newline='') as outfile:
    datawriter = csv.writer(outfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in validations_score:  
            new_row = list(row)
            datawriter.writerow(new_row)
        
        
print('Best # ' + str(best_model))
print('Best Validation score : ' + str(np.round(best_validation,4)))



    
 
