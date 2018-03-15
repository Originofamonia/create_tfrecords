'''
from http://blog.csdn.net/miaomiaoyuan/article/details/56865361
By Qiyuan An.

'''

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def create_tfrecords(cwd, filename):
    classes = ['Cup', 'FathersDay', 'Gun_like_objects', 'GunPart', 'Horse', 'Pistol', 'PortalGun', 'Revolver', 'SpecialRevolver', 'ToyGun']
    # classes = ['agunkeychain', 'assemble1', 'BananaGun', 'BerettaPropGun', 'BladeRunner', 'DesertEagleGun', 'mal_gunV2', 'MIB_GUN', 'px4', 'smith&wesson']  # 认为设置10类
    writer = tf.python_io.TFRecordWriter(cwd + filename)  # 要生成的文件
    index = 0
    while index < len(classes):
        class_path = cwd + '/' + classes[index] + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的地址
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
        index += 1
    writer.close()


def read_and_decode(filename):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的1通道图片
    img = tf.cast(img, tf.float32) * (1. / 255)  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label


def restore_tfrecords(cwd, path, num_of_images):
    filename_queue = tf.train.string_input_producer([path])  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )  # 将image数据和label取出来

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])  # reshape为128*128的3通道图片
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num_of_images):
            example, lbl = sess.run([image, label])  # 在会话中取出image和label
            img = Image.fromarray(example, mode='RGB')  # 这里Image是之前提到的
            img.save(cwd+str(i) + '_''Label_'+str(lbl) + '.png', format='PNG')  # 保存图片
        coord.request_stop()
        coord.join(threads)


def main(argv):
    cwd = './data/new_3ch_images/training-images/'
    tfr_filename = 'train_3ch.tfrecords'
    input_path = './data/new_3ch_images/training-images/train_3ch.tfrecords'
    train_num = 14280
    # create_tfrecords(cwd=cwd, filename=tfr_filename)
    restore_tfrecords(cwd=cwd, path=input_path, num_of_images=train_num)


if __name__ == '__main__':
    tf.app.run()
