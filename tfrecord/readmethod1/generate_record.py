#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def paths2list(path_file_name):
    list = []
    for line in open(path_file_name):
        list.append(line[0:len(line)-1])
    return list
def pathslabel2list(path_file_name):
    list = []
    for line in open(path_file_name):
        #存储是label是string格式，这里需要强转一下
        list.append(int(line[0:len(line)-1]))
    return list
def one_hot_2_int(one_hot):
    for i in range(10):
        if one_hot[i] == 1:
            return  i
        else:
            continue
    return 0
# train_image_list = paths2list(r"E:\mnist_jpg\jpg\train\train_image_list.txt")
# train_image_label_list =  pathslabel2list(r"E:\mnist_jpg\jpg\train\train_label_list.txt")

# 猫1狗0
cwd = os.getcwd()
print(cwd)

train_image_list = paths2list('train_image_list.csv')
train_image_label_list =  pathslabel2list('train_label_list.csv')


#定义创建TFRcord的文件

def image2tfrecord(image_list,label_list):
    len2 = len(image_list)
    print("len=",len2)
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for i in range(len2):
        #读取图片并解码
        image = Image.open(image_list[i])
        image = image.resize((400,400))
        #转化为原始字节
        image_bytes = image.tobytes()
        #创建字典
        features = {}
        #用bytes来存储image
        features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        # 用int64来表达label
        features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_list[i])]))
        #将所有的feature合成features
        tf_features = tf.train.Features(feature=features)
        #转成example
        tf_example = tf.train.Example(features=tf_features)
        #序列化样本
        tf_serialized = tf_example.SerializeToString()
        #将序列化的样本写入rfrecord
        writer.write(tf_serialized)
    writer.close()
#调用上述接口，将image与label数据转化为tfrecord格式的数据
image2tfrecord(train_image_list,train_image_label_list)

#定义解析数据函数
#入参example_proto也就是tf_serialized
def pares_tf(example_proto):
    #定义解析的字典
    dics = {}
    dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['image'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    #调用接口解析一行样本
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    image = tf.decode_raw(parsed_example['image'],out_type=tf.uint8)
    image = tf.reshape(image,shape=[400*400*3])
    #这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
    # 有了这里的归一化处理，精度与原始数据一致
    image = tf.cast(image,tf.float32)*(1./255)-0.5
    label = parsed_example['label']
    label = tf.cast(label,tf.int32)
    # label = tf.one_hot(label, depth=10, on_value=1)
    return image,label



dataset = tf.data.TFRecordDataset(filenames=['train.tfrecords'])
dataset = dataset.map(pares_tf)

dataset = dataset.shuffle(1000).batch(3).repeat(2)

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()



with tf.Session() as sess:

    i = 0
    try:
        while True:
            #通过session每次从数据集中取值
            image,label= sess.run(fetches=next_element)
            print(i)
            print(image)
            print(image.shape)
            print(label)
            i = i+1
            for j in np.arange(3):
                print('label: %d' % label[j])

                plt.imshow(image[j].reshape(400,400,3))
                plt.show()
            if(i == 6):
                print('----------------')



    except tf.errors.OutOfRangeError:
        print("end!")



