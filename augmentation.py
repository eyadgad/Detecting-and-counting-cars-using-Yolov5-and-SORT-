import os
import sys
import numpy as np
from PIL import Image
import cv2 
import albumentations as A
from albumentations.core.composition import BboxParams, Compose


def my_augmentation():
    transforms = A.Compose([
        # A.Normalize(),
        # A.Blur(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Downscale(p=0.3),
        # A.Superpixels(p=0.3),
        A.RandomContrast(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Sharpen(p = 0.5),
        # A.RGBShift(p=0.5),
        # A.RandomRain(p=0.3),
        # A.RandomFog(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return transforms

def get_yolo_labels(label_path):
    boxes = []
    label_file = open(label_path, 'r')
    for line in label_file:
        _, x_center, y_center, w, h = line.split(' ')
        boxes.append([float(x_center), float(y_center), float(w), float(h)])
    label_file.close()
    return boxes

def write_yolo_labels(label_path, boxes):
    label_file = open(label_path, 'w')
    for box in boxes:
        line = '0 ' + ' '.join([str(i) for i in box]) + '\n'
        label_file.write(line)
    label_file.close()

def denorm_yolo(boxes, image_shape):
    new_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box
        h = int(float(h) * image_shape[0])
        w = int(float(w) * image_shape[1])
        
        y_center = int(float(y_center) * image_shape[0])
        x_center = int(float(x_center) * image_shape[1])

        y = y_center - (h//2)
        x = x_center - (w//2)
        y2 = y + h
        x2 = x + w
        new_boxes.append([x, y, x2, y2])

    return new_boxes

    
def augment_yolo_data(input_data_dir,output_data_dir,class_list):
    data_dir = input_data_dir
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    image_paths = [os.path.join(images_dir, image) for image in sorted(os.listdir(images_dir))]
    label_paths = [os.path.join(labels_dir, label) for label in sorted(os.listdir(labels_dir))]

    new_images_dir = os.path.join(output_data_dir, 'images')
    new_labels_dir = os.path.join(output_data_dir, 'labels')
    
    transform = my_augmentation()
    class_labels = class_list
    counter = 0
    for image_path, label_path in zip(image_paths, label_paths):
        counter += 1

        image = cv2.imread(image_path)
        boxes = get_yolo_labels(label_path)

        cv2.imwrite(os.path.join(new_images_dir, str(counter))+'.jpg', image)
        write_yolo_labels(os.path.join(new_labels_dir, str(counter)+'.txt'), boxes)

        for _ in range(20):
            counter += 1
            transformed = transform(image = image, bboxes = boxes, class_labels = class_labels*len(boxes))
            cv2.imwrite(os.path.join(new_images_dir, str(counter))+'.jpg', transformed['image'])
            write_yolo_labels(os.path.join(new_labels_dir, str(counter)+'.txt'), transformed['bboxes'])

def draw_boxes(image, boxes):
    boxed_image = image.copy()
    for box in boxes:
        x, y, x2, y2 = box
        boxed_image = cv2.rectangle(boxed_image, (int(x), int(y)), (int(x2), int(y2)), (255,0,0), 2)
    return boxed_image

if __name__ == "__main__":
    input_data_dir='data'
    output_data_dir='new_data'
    class_list=['car']
    augment_yolo_data(input_data_dir,output_data_dir,class_list)