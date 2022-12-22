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


from tensorflow.keras.models import Model

model0 =tf.keras.models.load_model(model_path_baseline)  # include here your original model
model1 =tf.keras.models.load_model(model_path_harmonized)

layer_name = 'fc2'
intermediate_layer_model0 = Model(inputs=model0.input,
                                 outputs=model0.get_layer(layer_name).output)
intermediate_layer_model1 = Model(inputs=model1.input,
                                 outputs=model1.get_layer(layer_name).output)

def get_dataset(preprocess, batch_size, filter, shuffle=True):
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

  if (filter == 1) :
    dataset = dataset.filter(lambda filename, image, label: tf.math.argmax(label) < tf.constant(398, dtype=tf.int64))
  
  if (filter == 2) :
    dataset = dataset.filter(lambda filename, image, label: tf.math.argmax(label) > tf.constant(398, dtype=tf.int64))

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

folder_path = '/users/solaiya/controversial_images/sixteen_way_class'
os.mkdir(folder_path)

LIMIT = 50000
batch_size = 128

LIMIT_train = 2500
batch_size_animal = 100

ds_train = get_dataset(preprocess, batch_size = batch_size, filter= 0, shuffle= True)
ds = get_dataset(preprocess, batch_size = batch_size, filter= 0, shuffle= False)


predictions_array0_train = np.empty((0, 4096))
predictions_array1_train = np.empty((0, 4096))
filename_array_train = np.empty((0,))
Y_label_train = np.empty((0, 1000))


def get_category_ind(imagenet_label):
    human_categories = hm.get_human_object_recognition_categories()
    for cat_ind, category in enumerate(human_categories):
        Human_Cat = hm.HumanCategories()
        ind = Human_Cat.get_imagenet_indices_for_category(category)
        if imagenet_label in ind:
            return cat_ind
    return None


X = []
Y = []
F = []
batch_counter = 0

for batch in ds_train.take(LIMIT_train // batch_size_animal):
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
  preds1 = intermediate_layer_model1.predict(X, batch_size=None)
  predictions_array0_train = np.append(predictions_array0_train, preds0, axis = 0)
  predictions_array1_train = np.append(predictions_array1_train, preds1, axis = 0)
  filename_array_train = np.append(filename_array_train, F, axis = 0)


## get predictions for all 50000 images
predictions_array0 = np.empty((0, 4096))
predictions_array1 = np.empty((0, 4096))
filename_array = np.empty((0,))
Y_label = np.empty((0, 1000))

X = []
Y = []
F = []
batch_counter = 0

for batch in ds.take(LIMIT // batch_size):
  print("Processing batch", batch_counter)
  batch_counter += 1

  f, x, y = batch
  F = np.array(list(f.numpy()))
  X = np.array(list(x))
  Y = np.array(list(y))

  label_ind = np.argmax(Y, axis = 1)
  category_ind = [get_category_ind(ind) for ind in list(label_ind)]
  Y_label = np.append(Y_label, np.array(category_ind))
  # N = 0 OR 1, 0 for baseline, 1 for harmonized

  preds0 = intermediate_layer_model0.predict(X, batch_size=None)
  preds1 = intermediate_layer_model1.predict(X, batch_size=None)
  predictions_array0 = np.append(predictions_array0, preds0, axis = 0)
  predictions_array1 = np.append(predictions_array1, preds1, axis = 0)
  filename_array = np.append(filename_array, F, axis = 0)

import numpy as np
#Import svm model
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def linear_classifier(predictions_array_train, predictions_array_val, model_type):
  distance_to_decision_boundary_array = np.empty((0,16))

  Y_label_func_train = Y_label_train
  Y_label_func = Y_label
  predictions_array_train = predictions_array_train[Y_label_func_train != np.array(None)]
  Y_label_func_train = Y_label_func_train[Y_label_func_train != np.array(None)]
  Y_label_func_train=Y_label_func_train.astype('int') 

  X_train = predictions_array_train
  y_train = Y_label_func_train
  #Create a svm Classifier
  clf = OneVsRestClassifier(SVC()) # Linear Kernel
  #Train the model using the training sets
  clf.fit(X_train, y_train)
  
  predictions_array_val = predictions_array_val[Y_label_func != np.array(None)]
  Y_label_func = Y_label_func[Y_label_func != np.array(None)]
  Y_label_func=Y_label_func.astype('int') 

  distance_to_decision_boundary = clf.decision_function(predictions_array_val)
  distance_to_decision_boundary_array = np.concatenate(
    (distance_to_decision_boundary_array, distance_to_decision_boundary), axis=0)
  return distance_to_decision_boundary_array, Y_label_func

results0= linear_classifier(predictions_array0_train,predictions_array0, "baseline")
distance0= results0[0]
distance1= linear_classifier(predictions_array1_train,predictions_array1,"harmonized")[0]
Y_label_table = results0[1]

# plt.scatter(distance0,distance1)  
# plt.xlabel('Baseline Model Hyperplane Distance')
# plt.ylabel('Harmonized Model Hyperplane Distance')
# plt.savefig(
#   "/users/solaiya/controversial_images/abs_results_harmonized/abs_hyperplane_dist_val.png", bbox_inches='tight')  

# harmonized_distance_diff = abs(distance1 - distance0)
# baseline_distance_diff = abs(distance0 - distance1)

# abs_harmonized = abs(distance1)
# abs_baseline = abs(distance0)

# harmonized_distance_diff = []
# baseline_distance_diff = []


# for d_ind, d in enumerate(distance1): 
#     ## if both harmonized and unharmonized models are correct
#     if (((d > 0 and Y_animal_label[d_ind] == 1) and (distance0[d_ind] > 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d < 0 and Y_animal_label[d_ind] == 0) and (distance0[d_ind] < 0 and Y_animal_label[d_ind] == 0))):
#         harmonized_distance_diff.append(abs_harmonized[d_ind] - abs_baseline[d_ind])

#     ## if harmonized is correct and baseline is wrong:
#     elif (((d > 0 and Y_animal_label[d_ind] == 1) and (distance0[d_ind] < 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d < 0 and Y_animal_label[d_ind] == 0) and (distance0[d_ind] > 0 and Y_animal_label[d_ind] == 0))):
#         harmonized_distance_diff.append(abs_harmonized[d_ind] - (abs_baseline[d_ind] *-1))

#     ## if harmonized is wrong and baseline is correct:
#     elif (((d < 0 and Y_animal_label[d_ind] == 1) and (distance0[d_ind] > 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d > 0 and Y_animal_label[d_ind] == 0) and (distance0[d_ind] < 0 and Y_animal_label[d_ind] == 0))):
#         harmonized_distance_diff.append((abs_harmonized[d_ind] * -1) - abs_baseline[d_ind])
    
#     else: ## if both are wrong
#       if abs_harmonized[d_ind] > abs_baseline[d_ind]:
#         harmonized_distance_diff.append((abs_harmonized[d_ind] - abs_baseline[d_ind]) * -1)
#       else:
#         harmonized_distance_diff.append(abs_harmonized[d_ind] - abs_baseline[d_ind])


# for d_ind, d in enumerate(distance0): 
#     ## if both harmonized and unharmonized models are correct
#     if (((d > 0 and Y_animal_label[d_ind] == 1) and (distance1[d_ind] > 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d < 0 and Y_animal_label[d_ind] == 0) and (distance1[d_ind] < 0 and Y_animal_label[d_ind] == 0))):
#         baseline_distance_diff.append(abs_baseline[d_ind] - abs_harmonized[d_ind])

#     ## if harmonized is wrong and baseline is correct:
#     elif (((d > 0 and Y_animal_label[d_ind] == 1) and (distance1[d_ind] < 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d < 0 and Y_animal_label[d_ind] == 0) and (distance1[d_ind] > 0 and Y_animal_label[d_ind] == 0))):
#         baseline_distance_diff.append(abs_baseline[d_ind] - (abs_harmonized[d_ind] *-1))

#     ## if harmonized is correct and baseline is wrong:
#     elif (((d < 0 and Y_animal_label[d_ind] == 1) and (distance1[d_ind] > 0 and Y_animal_label[d_ind] == 1)) or 
#     ((d > 0 and Y_animal_label[d_ind] == 0) and (distance1[d_ind] < 0 and Y_animal_label[d_ind] == 0))):
#         baseline_distance_diff.append((abs_baseline[d_ind] * -1) - abs_harmonized[d_ind])
    
#     else: ## if both are wrong
#       if abs_baseline[d_ind] > abs_harmonized[d_ind]:
#         baseline_distance_diff.append((abs_baseline[d_ind] - abs_harmonized[d_ind]) * -1)
#       else:
#         baseline_distance_diff.append(abs_baseline[d_ind] - abs_harmonized[d_ind])


# harmonized_distance_diff = np.array(harmonized_distance_diff)
# baseline_distance_diff = np.array(baseline_distance_diff)

# largest_harmonized_ids = harmonized_distance_diff.argsort()[::-1][:15]
# largest_baseline_ids = baseline_distance_diff.argsort()[::-1][:15]
 
# import skimage.io
# import skimage.util
# import shutil
# import cv2

# model_types = ['harmonized', 'baseline']
# difference_ids = [largest_harmonized_ids, largest_baseline_ids]
# for ind, model_type in enumerate(model_types):
#     image_path = '/users/solaiya/controversial_images/abs_results_harmonized/abs_' + model_type + '_better_images'
#     os.mkdir(image_path)

#     image_montage = []
#     for image_id in difference_ids[ind]:
#         rough_filepath = str(filename_array[image_id])
#         filepath = rough_filepath[2:]
#         actual_filepath = filepath[:-1]
#         original = r'/gpfs/data/tserre/irodri15/DATA/ILSVRC/Data/CLS-LOC/val/' + actual_filepath
#         target = r'/users/solaiya/controversial_images/abs_'+ model_type + '_better_images/' + actual_filepath.replace("/", "_")
#         shutil.copyfile(original, target)

#         a = skimage.io.imread(target)
#         a = cv2.resize(a, dsize=(600, 600), interpolation=cv2.INTER_CUBIC) 
#         image_montage.append(a)
#         print(a.shape)

#     m = skimage.util.montage(image_montage, grid_shape = (3, 5), multichannel=True)

#     skimage.io.imsave(image_path + '/skimage_montage_'+ model_type+ '.jpg', m)



# convert array into dataframe
image_margin_df = pd.DataFrame(
    {"filepaths" : filename_array[Y_label != np.array(None)], "Category" : Y_label_table }
    )

baseline_dist_df = pd.DataFrame(distance0)
harmonized_dist_df = pd.DataFrame(distance1)

header_list= hm.get_human_object_recognition_categories()


baseline_dist_df.to_csv('/users/solaiya/controversial_images/sixteen_way_class/baseline_classifier_margins_2500.csv', 
index=False, header= header_list)

harmonized_dist_df.to_csv('/users/solaiya/controversial_images/sixteen_way_class/harmonized_classifier_margins_2500.csv', 
index=False, header= header_list)


 
# save the dataframe as a csv file
image_margin_df.to_csv("/users/solaiya/controversial_images/sixteen_way_class/extra_info.csv", index=False)

image_filename_df = pd.DataFrame({"Filepaths" : filename_array_train})
# save the dataframe as a csv file
image_filename_df.to_csv("/users/solaiya/controversial_images/sixteen_way_class/training_filepaths.csv", index=False)
