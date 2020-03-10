import struct

import cv2
import numpy as np

"""
Source:
https://gist.github.com/ceykmc/c6f3d27bb0b406e91c27
"""


def extract_labels(dataset_label_file_path, label_file_path):
    with open(dataset_label_file_path, "rb") as dataset_label_file:
        # 32 bit integer magic number
        dataset_label_file.read(4)
        # 32 bit integer number of items
        dataset_label_file.read(4)
        # actual test label
        label_file = open(label_file_path, "w")
        label = dataset_label_file.read(1)
        while label:
            label_file.writelines(str(label[0]) + "\n")
            label = dataset_label_file.read(1)
        label_file.close()


def extract_images(images_file_path, images_save_folder, dataset_label_file_path):
    with open(dataset_label_file_path, "rb") as labelFile:
        temp = labelFile.readlines()
        with open(images_file_path, "rb") as images_file:
            # 32 bit integer magic number
            images_file.read(4)
            # 32 bit integer number of images
            images_file.read(4)
            # 32 bit number of rows
            images_file.read(4)
            # 32 bit number of columns
            images_file.read(4)
            # every image contain 28 x 28 = 784 byte, so read 784 bytes each time
            count = 1
            image = np.zeros((28, 28, 1), np.uint8)
            image_bytes = images_file.read(784)
            while image_bytes:
                image_unsigned_char = struct.unpack("=784B", image_bytes)
                for i in range(784):
                    image.itemset(i, image_unsigned_char[i])
                image_save_path = "%s/Image_%d_Label_%s.png" % (
                    images_save_folder, count, temp[count - 1].decode("utf-8")[0])
                cv2.imwrite(image_save_path, image)
                image_bytes = images_file.read(784)
                count += 1
            print("Finished Extracting Images from {} \n "
                  "TO: {}".format(images_file_path.split("/")[-1], images_save_folder.split("/")[-1]))
