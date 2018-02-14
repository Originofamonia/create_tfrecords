# from http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

from random import shuffle
import glob
import numpy as np
import cv2
import tensorflow as tf
import sys

# 1: List images and label them
shuffle_data = True  # shuffle the addresses before saving
cat_dog_train_path = './train/*.jpg'

# read addresses and labels from the train folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = cat, 1 = dog

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0: int(0.6*len(addrs))]
train_labels = labels[0: int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)): int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)): int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(addrs)):]


# 2. A function to load images
def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


# 3. Convert data to features
'''
    Before we can store the data into a TFRecords file, we should stuff it in a protocol buffer called Example. Then, we serialize the protocol buffer to a string and write it to a TFRecords file. Example protocol buffer contains Features. Feature is a protocol to describe the data and could have three types: bytes, float, and int64. In summary, to store your data you need to follow these steps:

Open a TFRecords file using tf.python_io.TFRecordWriter
Convert your data into the proper data type of the feature using tf.train.Int64List, tf.train.BytesList, or  tf.train.FloatList
Create a feature using tf.train.Feature and pass the converted data to it
Create an Example protocol buffer using tf.train.Example and pass the feature to it
Serialize the Example to string using example.SerializeToString()
Write the serialized example to TFRecords file using writer.write
'''


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 4. Write data into a TFRecord file
train_filename = 'train.tfrecords'  # address to save the TFRecords

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()

    # load the image
    img = load_image(train_addrs[i])

    label = train_labels[i]

    # create a feature
    feature = {
        'train/label': _int64_feature(label),
        'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
    }
    # create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()


# 5. Write validation and test data into a TFRecord file
# open the TFRecords file
val_filename = 'val.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Val data: {}/{}'.format(i, len(val_addrs)))
        sys.stdout.flush()

    # load the image
    img = load_image(val_addrs[i])

    label = val_labels[i]

    # create a feature
    feature = {
        'val/label': _int64_feature(label),
        'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
    }

    # create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# open the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
        sys.stdout.flush()

    # load the image
    img = load_image(test_addrs[i])

    label = test_labels[i]

    # create a feature
    feature = {
        'test/label': _int64_feature(label),
        'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
    }

    # create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
