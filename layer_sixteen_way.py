import os
import glob
import os
import pandas as pd
import tensorflow as tf
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
import math as math
import human_categories as hm

im_path = '/gpfs/data/tserre/irodri15/DATA/ILSVRC/Data/CLS-LOC/val/'
model_path_baseline = '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/vgg_baseline.h5'
model_path_harmonized = '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/vgg_frosty_eon.h5'
# model_path_baseline = '/Users/stephanie/Desktop/Serre Lab/model_human_comparison/model-vs-human/vgg_baseline.h5'
# model_path_harmonized = '/Users/stephanie/Desktop/Serre Lab/model_human_comparison/model-vs-human/vgg_frosty_eon.h5'
test_images_path = '/users/solaiya/controversial_images/model_human_comparison/model-vs-human/datasets/colour/dnn/session-1/'
categories = hm.get_human_object_recognition_categories()


from tensorflow.keras.models import Model

model0 =tf.keras.models.load_model(model_path_baseline)  # include here your original model
model1 =tf.keras.models.load_model(model_path_harmonized)

# model_layers = []

# for layer in model0.layers:
#     model_layers.append(layer.name)

# model_layers = ['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 
# 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 
# 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 
# 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 
# 'flatten', 'fc1', 'fc2', 'predictions']


model_layers = ['fc1', 'fc2']

def get_dataset(preprocess, batch_size, shuffle=True):
  filenames = []
  folders = os.listdir(im_path)
  folders.sort()
  for s in folders:
    new_list = os.listdir(im_path + s + '/')
    new_list = [s + "/" + r for r in new_list]
    filenames = filenames + new_list
  filenames = np.array(np.sort(filenames))[:50000]
  
  '''
  r = list(filter(lambda f: ' (1)' in f, filenames))
  print(r)
  '''

  labels = np.array([folders.index(s.split('/')[0]) for s in filenames])

  final_ids = np.arange(len(filenames))
  
  if shuffle:
    # Random shuffle. Evenly distribute classes!
    ids = np.arange(len(filenames))
    for i in range(1000):
      temp = ids[i*50:(i + 1)*50]
      np.random.shuffle(temp)
      ids[i*50:(i + 1)*50] = temp
    
    for i in range(50):
      temp = []
      for j in range(1000): 
        temp.append(ids[j*50 + i]) # Add one of each class

      # Shuffle within each group of 1000
      shuffler = np.arange(len(temp))
      np.random.shuffle(shuffler)
      temp = np.array(temp)
      temp = temp[shuffler]

      # Add group of 1000 to final IDs
      final_ids[i*1000:(i + 1)*1000] = temp

  dataset = tf.data.Dataset.from_tensor_slices((filenames[final_ids], labels[final_ids]))
  # for element in dataset:
  #   print(element[1].numpy())

  dataset = dataset.map(_parse_element, num_parallel_calls=AUTO)

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda f, x, y : (f, preprocess(x), y), num_parallel_calls=AUTO)
  dataset = dataset.prefetch(AUTO)

  return dataset


def get_human_dataset(preprocess, batch_size, shuffle=True):
  filenames = []
  images = os.listdir(test_images_path)
  for s in images:
    filenames.append(s)
  filenames = np.array(np.sort(filenames))[:1280]
  
  '''
  r = list(filter(lambda f: ' (1)' in f, filenames))
  print(r)
  '''

  labels_list = []
  for s in filenames: 
    for cat in categories:
        if cat in s: 
            labels_list.append(categories.index(cat))


  labels = np.array(labels_list)

  final_ids = np.arange(len(filenames))

  dataset = tf.data.Dataset.from_tensor_slices((filenames[final_ids], labels[final_ids]))

  dataset = dataset.map(_parse_element_test, num_parallel_calls=AUTO)

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda f, x, y : (f, preprocess(x), y), num_parallel_calls=AUTO)
  dataset = dataset.prefetch(AUTO)

  return dataset

MEAN_IMAGENET = tf.constant([0.485, 0.456, 0.406], shape=[3], dtype=tf.float32)
STD_IMAGENET  =  tf.constant([0.229, 0.224, 0.225], shape=[3], dtype=tf.float32)

DIVISOR = tf.cast(1.0 / 255.0, tf.float32)
STD_DIVISOR = tf.cast(1.0 / STD_IMAGENET, tf.float32)

def preprocess(image):
  image = tf.cast(image, tf.float32)

  image = image * DIVISOR
  image = image - MEAN_IMAGENET
  image = image * STD_DIVISOR

  return image


AUTO = tf.data.AUTOTUNE
SIZE = 224

def _parse_element(filename, label):
  path = tf.strings.reduce_join([im_path,filename])

  image_string = tf.io.read_file(path)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.cast(image_decoded, tf.float32)
  #image = tf.image.resize(image, (224, 224))
  image = tf.image.resize_with_crop_or_pad(image, 224, 224)


  label = tf.one_hot(label, 1000)

  return filename, image, label

def _parse_element_test(filename, label):
  path = tf.strings.reduce_join([test_images_path,filename])

  image_string = tf.io.read_file(path)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.cast(image_decoded, tf.float32)
  #image = tf.image.resize(image, (224, 224))
  image = tf.image.resize_with_crop_or_pad(image, 224, 224)


  label = tf.one_hot(label, 1000)

  return filename, image, label

# folder_path = '/users/solaiya/controversial_images/layers_sixteen_way_class'
# os.mkdir(folder_path)

LIMIT_test = 1280
batch_size = 128

LIMIT_train = 50000


ds_train = get_dataset(preprocess, batch_size = batch_size, shuffle= True)
ds_test = get_human_dataset(preprocess, batch_size = batch_size, shuffle= False)



def get_category_ind(imagenet_label):
    human_categories = hm.get_human_object_recognition_categories()
    for cat_ind, category in enumerate(human_categories):
        Human_Cat = hm.HumanCategories()
        ind = Human_Cat.get_imagenet_indices_for_category(category)
        if imagenet_label in ind:
            return cat_ind
    return None

def linear_classifier(predictions_array_train, predictions_array_test, Y_label_train, Y_label_test):
  distance_to_decision_boundary_array = np.empty((0,16))

  Y_label_func_train = Y_label_train
  Y_label_func_test = Y_label_test
  predictions_array_train = predictions_array_train[Y_label_func_train != np.array(None)]
  Y_label_func_train = Y_label_func_train[Y_label_func_train != np.array(None)]
  Y_label_func_train=Y_label_func_train.astype('int') 

  X_train = predictions_array_train
  y_train = Y_label_func_train
  #Create a svm Classifier
  clf = OneVsRestClassifier(SVC()) # Linear Kernel
  #Train the model using the training sets
  clf.fit(X_train, y_train)
  
  Y_label_func_test=Y_label_func_test.astype('int') 

  distance_to_decision_boundary = clf.decision_function(predictions_array_test)
  distance_to_decision_boundary_array = np.concatenate(
    (distance_to_decision_boundary_array, distance_to_decision_boundary), axis=0)
  return distance_to_decision_boundary_array, Y_label_func_test

import numpy as np
#Import svm model
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

layer_name = 'fc2'
for layer_name in model_layers:
    intermediate_layer_model0 = Model(inputs=model0.input,
                                    outputs=model0.get_layer(layer_name).output)
    intermediate_layer_model1 = Model(inputs=model1.input,
                                    outputs=model1.get_layer(layer_name).output)

                                
    X = []
    Y = []
    F = []
    batch_counter = 0

    predictions_array0_train = np.empty((0, 4096))
    predictions_array1_train = np.empty((0, 4096))
    filename_array_train = np.empty((0,))
    Y_label_train = np.empty((0, 1000))

    for batch in ds_train.take(LIMIT_train // batch_size):
        print("Processing batch", batch_counter)
        batch_counter += 1

        f, x, y = batch
        F = np.array(list(f.numpy()))
        X = np.array(list(x))
        Y = np.array(list(y))

        label_ind = np.argmax(Y, axis = 1)
        category_ind = [get_category_ind(ind) for ind in list(label_ind)]
        Y_label_train = np.append(Y_label_train, np.array(category_ind))
        # N = 0 OR 1, 0 for baseline, 1 for harmonized

        preds0 = intermediate_layer_model0.predict(X, batch_size=None)
        print(predictions_array0_train.shape)
        print(preds0.shape)
        preds1 = intermediate_layer_model1.predict(X, batch_size=None)
        predictions_array0_train = np.append(predictions_array0_train, preds0, axis = 0)
        predictions_array1_train = np.append(predictions_array1_train, preds1, axis = 0)
        filename_array_train = np.append(filename_array_train, F, axis = 0)


    ## get predictions for all test images
    predictions_array0_test = np.empty((0, 4096))
    predictions_array1_test = np.empty((0, 4096))
    filename_array_test = np.empty((0,))
    Y_label_test= np.empty((0, 16))

    X = []
    Y = []
    F = []
    batch_counter = 0

    for batch in ds_test.take(LIMIT_test // batch_size):
        print("Processing batch", batch_counter)
        batch_counter += 1

        f, x, y = batch
        F = np.array(list(f.numpy()))
        X = np.array(list(x))
        Y = np.array(list(y))
        
        label_ind = np.argmax(Y, axis = 1)
        Y_label_test = np.append(Y_label_test, np.array(label_ind))
        # N = 0 OR 1, 0 for baseline, 1 for harmonized

        preds0 = intermediate_layer_model0.predict(X, batch_size=None)
        preds1 = intermediate_layer_model1.predict(X, batch_size=None)
        predictions_array0_test = np.append(predictions_array0_test, preds0, axis = 0)
        predictions_array1_test = np.append(predictions_array1_test, preds1, axis = 0)
        filename_array_test = np.append(filename_array_test, F, axis = 0)


    results0= linear_classifier(predictions_array0_train, predictions_array0_test, Y_label_train, Y_label_test )
    distance0= results0[0]
    Y_label_table = results0[1]
    distance1= linear_classifier(predictions_array1_train, predictions_array1_test, Y_label_train, Y_label_test)[0]



# convert array into dataframe
# image_margin_df = pd.DataFrame(
#     {"filepaths" : filename_array_test, "Category" : Y_label_table }
#     )

    image_margin_df = pd.DataFrame({"Categories": Y_label_table})
    baseline_dist_df = pd.DataFrame(distance0)
    harmonized_dist_df = pd.DataFrame(distance1)
    header_list= hm.get_human_object_recognition_categories()


    baseline_dist_df.to_csv('/users/solaiya/controversial_images/layers_sixteen_way_class/baseline_classifier_margins_'+layer_name+'.csv', 
    index=False, header= header_list)

    harmonized_dist_df.to_csv('/users/solaiya/controversial_images/layers_sixteen_way_class/harmonized_classifier_margins_'+layer_name+'.csv', 
    index=False, header= header_list)


# save the dataframe as a csv file
image_margin_df.to_csv("/users/solaiya/controversial_images/layers_sixteen_way_class/test_category_info.csv", index=False)

image_filename_df = pd.DataFrame({"Filepaths" : filename_array_train})
# save the dataframe as a csv file
image_filename_df.to_csv("/users/solaiya/controversial_images/layers_sixteen_way_class/training_filepaths.csv", index=False)
