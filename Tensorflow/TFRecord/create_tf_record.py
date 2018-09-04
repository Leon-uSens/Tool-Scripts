import sys
import os
import numpy as np
import tensorflow as tf

from tf_example_util import create_tf_example
from datetime import datetime
from sklearn.model_selection import train_test_split

train_data_path = 'E:/TensorflowProjects/tank/data/images/'            
train_record_output_path = 'E:/TensorflowProjects/tank/data/records/tank_train.record'   
test_record_output_path = 'E:/TensorflowProjects/tank/data/records/tank_test.record'    

default_image_format = '.JPEG'
default_annotation_format = '.txt'

test_set_split_ratio = 0.2

def main(_):   
    data_files = os.listdir(train_data_path)
    image_list = []
    annotation_list = []

    # Append images and annotations into lists.
    for data_file in data_files:
        file_name = os.path.splitext(data_file)[0]
        file_extension = os.path.splitext(data_file)[1]

        if file_extension == default_image_format:
            image_list.append(file_name)
        elif file_extension == default_annotation_format:
            annotation_list.append(file_name)
          
    # Sort the image and the annotation lists.
    image_list.sort()
    annotation_list.sort()
    image_list = np.array(image_list)
    annotation_list = np.array(annotation_list)

    # Split the data set into training set and test set.
    train_set, test_set, _, _ = train_test_split(image_list, annotation_list, test_size = test_set_split_ratio, random_state = 43)

    # Write the training tf record.
    train_record_writer = tf.python_io.TFRecordWriter(train_record_output_path)
    train_counter = 0
    for train_data_name in train_set:
        tf_example = create_tf_example(train_data_path, train_data_name, default_image_format, default_annotation_format)
        train_record_writer.write(tf_example.SerializeToString())

        train_counter = train_counter + 1
        if train_counter % 1000 == 0:
            print('Created %d training samples in training record of %d training samples.' % (train_counter, len(train_set)))
    train_record_writer.close()

    # Write the test tf record.
    test_record_writer = tf.python_io.TFRecordWriter(test_record_output_path)
    test_counter = 0
    for test_data_name in test_set:       
        tf_example = create_tf_example(train_data_path, test_data_name, default_image_format, default_annotation_format)
        test_record_writer.write(tf_example.SerializeToString())

        test_counter = test_counter + 1
        if test_counter % 1000 == 0:
            print('Created %d test samples in test record of %d test samples.' % (test_counter, len(test_set)))
    test_record_writer.close()

if __name__ == '__main__':
    tf.app.run()