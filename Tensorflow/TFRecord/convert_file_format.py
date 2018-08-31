import os
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import cv2
from glob import glob

source_path = 'E:/TensorflowProjects/tank/train_data/annotations/'
target_path = 'E:/TensorflowProjects/tank/train_data/images/'

default_source_format = '.xml'
default_target_format = '.txt'
coding_format = "utf8"

label_map = {'Yellow_Tank': '1'}

def convert_xml_to_txt(source_file, target_path):
    """
    Function to convert the xml file to a txt file.

    args: 
        source_file: source xml file;.
        target_path: target txt file path.
    """

    assert source_file.endswith(default_source_format), "Unsupported format." 
    # Create the xml parser.
    xml_parser = etree.XMLParser(encoding = coding_format)
    # Create the xml tree with the parser.
    xml_tree = ElementTree.parse(source_file, parser = xml_parser).getroot()

    # Read image size.
    image_size = xml_tree.find('size')
    image_width = float(image_size.find('width').text)
    image_height = float(image_size.find('height').text)

    # Create and open the target file.
    source_file_name = os.path.splitext(source_file)[0]
    target_file = source_file_name + default_target_format
    os.chdir(target_path)
    with open(target_file, 'w') as target:
        for object_iterator in xml_tree.findall('object'):
                # Write the object label into the target file.
                label = object_iterator.find('name').text
                assert label in label_map, "Undefined label."
                target.write(label_map[label] + ' ')

                # Write the bounding box parameters into the target file.
                bounding_box = object_iterator.find("bndbox")
                xmin = float(bounding_box.find('xmin').text) / image_width
                ymin = float(bounding_box.find('ymin').text) / image_height
                xmax = float(bounding_box.find('xmax').text) / image_width               
                ymax = float(bounding_box.find('ymax').text) / image_height
                
                # Warn if there are any incorrect bounding box parameters.
                if xmin > xmax or ymin > ymax:
                    print('Fatal error: incorrect bounding box parameters in ' + source_file_name) 

                target.write(str(xmin) + ' ')
                target.write(str(ymin) + ' ')
                target.write(str(xmax) + ' ')
                target.write(str(ymax) + '\n')
    target.close()

if __name__ == '__main__':
    for source_file in os.listdir(source_path):
        os.chdir(source_path)
        convert_xml_to_txt(source_file, target_path)