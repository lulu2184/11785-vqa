import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import h5py
import os
# import _pickle as cPickle
import cPickle

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
train_indices_file = 'data/train36_imgid2idx.pkl'
val_indices_file = 'data/val36_imgid2idx.pkl'
train_folder = 'data/train2014'
val_folder = 'data/val2014'

feature_length = 2048
num_boxes = 36

def create_datasets(h5py_file, num_imgs):
    return (h5py_file.create_dataset('image_features', (num_imgs, num_boxes, feature_length), 'f'),
        h5py_file.create_dataset('image_bb', (num_imgs, num_boxes, 4), 'f'),
        h5py_file.create_dataset('spatial_features', (num_imgs, num_boxes, 6), 'f'))

def insert_data(ids, image_id, indices, bbs, bboxes, img_features, img_feature, spatial_features, spatial_feature, counter):
    ids.remove(image_id)
    indices[image_id] = counter
    bbs[counter, :, :] = bboxes
    img_features[counter, :, :] = img_feature
    spatial_features[counter, :, :] = spatial_feature

def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs

def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids

if __name__ == '__main__':
    train_h5py = h5py.File('data/train36.hdf5', 'w')
    val_h5py = h5py.File('data/val36.hdf5', 'w')
    train_indices = {}
    val_indices = {}

    print('loading imageid...')
    train_ids = load_imageid('data/train2014')
    val_ids = load_imageid('data/val2014')

    train_features, train_bb, train_spatial_features = create_datasets(train_h5py, len(train_ids))
    val_features, val_bb, val_spatial_features = create_datasets(val_h5py, len(val_ids))

    train_cnt = 0
    val_cnt = 0

    print('parsing tsv...')
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            image_id = int(item['image_id'])
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            bboxes = item['boxes']

            # bboxes: (x0, y0, x1, y1)
            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_feature = np.concatenate((scaled_x, scaled_y, scaled_width + scaled_x,
                scaled_height + scaled_y, scaled_width, scaled_height), axis=1)

            if image_id in train_ids:
                insert_data(train_ids, image_id, train_indices, train_bb, bboxes,
                    train_features, item['features'], train_spatial_features, spatial_feature, train_cnt)
                train_cnt += 1
            elif image_id in val_ids:
                insert_data(val_ids, image_id, val_indices, val_bb, bboxes,
                    val_features, item['features'], val_spatial_features, spatial_feature, val_cnt)
                val_cnt += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

        if len(train_ids) != 0:
            print('Warning: train_image_ids is not empty')

        if len(val_ids) != 0:
            print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    train_h5py.close()
    val_h5py.close()
    print("done!")
