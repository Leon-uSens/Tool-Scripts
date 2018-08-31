import tensorflow as tf
import cv2
# Import from PYTHONPATH: tensorflow/models/
from object_detection.utils import dataset_util

label_map = {'1': 'tank'}

def create_tf_example(data_path, data_name, image_format, annotation_format):
    """
    Function to convert the input raw data to tf_record format.

    args: 
        data_path: path of the data file.
        data_name: name of the data file.
        image_format:the extension of the image file.
        annotation_format: the extension of the annotation file.

    return: tensorflow example.
    """

    image_data = data_path + data_name + image_format
    annotation_data = data_path + data_name + annotation_format

    # Encode the image with the tensorflow API.
    with tf.gfile.GFile(image_data, 'rb') as fid:
        encoded_image = fid.read()

    # Write the 3D matrix information of the image into image_matrix.
    image_matrix = cv2.imread(image_data)
    height, width, channels = image_matrix.shape

    # Filename of the image. Empty if image is not from file.
    file_name = data_name.encode()  

    # List of normalized left x coordinates in bounding box (1 per box).
    xmins = []
    # List of normalized right x coordinates in bounding box (1 per box).
    xmaxs = []  

    # List of normalized top y coordinates in bounding box (1 per box).
    ymins = []
    # List of normalized bottom y coordinates in bounding box (1 per box).  
    ymaxs = []  

    # List of class name texts of bounding box (1 per box).
    class_texts = []
    # List of class name labels (integer) of bounding box (1 per box).  
    class_labels = []  

    with open(annotation_data, 'r') as annotation_file:
        lines = annotation_file.readlines()

        for line in lines:
            # Skip the blank lines.
            if len(line.split()) == 0: 
                continue

            # Split each line of the annotation_file to get the information of each bounding box.
            category, xmin, ymin, xmax, ymax = line.split()

            # Append the information of the bounding box to arrays.
            xmins.append(float(xmin))
            xmaxs.append(float(xmax))
            ymins.append(float(ymin))
            ymaxs.append(float(ymax))

            class_texts.append(label_map.get(category).encode())
            class_labels.append(int(category))

    # Create the tensorflow example.
    example = tf.train.Example(features = tf.train.Features(feature = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format.encode()),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_texts),
        'image/object/class/label': dataset_util.int64_list_feature(class_labels),
    }))

    return example