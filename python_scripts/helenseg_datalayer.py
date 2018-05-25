import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random
import os
import cv2
from skimage import transform as trans
import csv

x_0 = [41.9891, 84.7495, 63.7709, 45.8549, 83.6355]
y_0 = [40.9533, 39.5924, 65.5459, 87.4031, 86.5053]
src = np.array( (x_0, y_0) ).transpose([1,0]).astype(np.float32)

def Affinemat(angle, sx, sy, center = None, new_center = None):
    #angle = -angle/180.0*math.pi
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    if center is None:
        x = 0
        y = 0
    else:
        x = center[0]
        y = center[1]
    if new_center is None:
        nx = 0
        ny = 0
    else:
        nx = new_center[0]
        ny = new_center[1]
    a = cosine / sx
    b = sine / sx
    c = x-nx*a-ny*b
    d = -sine / sy
    e = cosine /sy
    f = y-nx*d-ny*e
    return (a,b,c,d,e,f)

def transform(x, y, v, img, imgSize, augmentation=True):
    src_r = src[np.nonzero(np.array(v)>0.5)]
    img2 = img.copy()
    dst = np.array( (x, y) ).transpose([1,0]).astype(np.float32)
    dst_r = dst[np.nonzero(np.array(v)>0.5)]
    tform = trans.SimilarityTransform()
    tform.estimate(dst_r, src_r)
    M = tform.params[0:2,:]
    out = cv2.warpAffine(img,M,(imgSize[1], imgSize[0]), borderValue = 0.0)
    # scipy.io.savemat('out1.png',dict(out=out))
    if augmentation:
        rate = (np.random.rand(1)-0.5)/10.0
        shift = np.minimum(imgSize[1]/2,imgSize[0]/2) * rate
        scale = 1+(np.random.rand(1)-0.5) / 10.0
        angle = (np.random.rand(1)-0.5)*(15.0/180.0)*math.pi
        mat = trans.SimilarityTransform(scale = scale, rotation=angle, translation=(shift,shift))
        M2 = mat.params[0:2,:]
        out = cv2.warpAffine(out, M2, (imgSize[1], imgSize[0]), borderValue = 0.0)
        # scipy.io.savemat('out2.png',dict(out=out))

    return out

    

def cls_sample(label):
    w, h, c = label.shape
    sele = int(w * h * 0.3)
    mask = np.zeros((w,h))
    for ii in range(c):
        lb = label[:,:,ii]
        lb_nz = np.sum(lb.flatten())
        if lb_nz < sele:
            mask += lb
        else:
            valid_all = int((sele / lb_nz) * (w * h))
            valid_ = np.random.randint(0,128*128,valid_all)
            valid_y, valid_x = np.unravel_index(valid_, (w, h))
            lbs = np.zeros_like(lb)
            lbs[valid_y, valid_x] = lb[valid_y, valid_x]
            mask += lbs
    return mask

class FacerealDatalayer(caffe.Layer):
    """
    Augmentation including:
    1. similarity transformation
    3. rand crop accoding to label
    4. (+ and - in tmporal
    axis) rand flip of 1st and 2nd image
    5. random 1-inter_step jump of frames

    - taking one image
    - apply superpixel
    - randomly set 4 (?) points in each label, avg their color
    - random crop 4, augmentation each
    - generate sparse plate, mask

    spn input: sparse plate
    guide input: y, mask
    gt: color channels
    """
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.root_dir = params['root']
        self.shape = np.array(params['shape'])

        if len(top) != 2:
            raise Exception("Need to define two tops: \
        data [first y, second y, first cbcr], label [second cbcr]")
        if len(bottom)!=0:
            raise Exception("Do not define a bottom.")

        split_im = os.path.join(self.root_dir, 'helen/train_dgx','train_im.txt')
        self.img_list = open(split_im, 'r').read().splitlines()
        split_lb = os.path.join(self.root_dir, 'helen/train_dgx','train_lb.txt')
        self.lab_list = open(split_lb, 'r').read().splitlines()

        self.idx = np.array(range(0, self.shape[0]))
        print(len(self.img_list))
        for id in range(self.shape[0]):
            self.idx[id] = random.randint(0, len(self.img_list)-1)

    def reshape(self, bottom, top):
        self.data = np.zeros((self.shape[0], self.shape[1], self.shape[2], self.shape[3]))
        self.label = np.zeros((self.shape[0], 11, self.shape[2], self.shape[3]))
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)

    def forward(self, bottom, top):
        for id in range(self.shape[0]):
            image, label = self.load_imagelabel_ac(self.idx[id])
            self.data[id] = image
            self.label[id] = label
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        for id in range(self.shape[0]):
            self.idx[id] = random.randint(0, len(self.img_list)-1)

    def backward(self, top, propagate_down, bottom):
        pass

    def load_imagelabel_ac(self, id):
        # print(id)
        im_name = os.path.join(self.root_dir, self.img_list[id])
        img = cv2.imread(im_name)
        lb_root = os.path.join(self.root_dir, self.lab_list[id])
        # get lb list
        for lb in range(11):
            name_ = lb_root.rpartition('/')
            name_ = name_[-1]
            lb_name = os.path.join(lb_root, name_+"_lbl{:02}.png".format(lb))
            lab_ = np.array(cv2.imread(lb_name, cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0
            if lb == 0:
                lab = lab_[:,:,np.newaxis]
            else:
                lab = np.append(lab, lab_[:,:,np.newaxis], axis=2)
        trans_ = np.append(img, lab, axis=2)
        # no v
        csv_name = im_name[:-4] + ".lmk"
        x_ = np.zeros(5)
        y_ = np.zeros(5)
        v_ = np.ones(5)
        fout = open(csv_name, 'r')
        read_data = fout.read().splitlines()
        count = 0
        for ii in read_data:
            read = ii.partition(' ')
            x_[count] = np.array(read[0], dtype=np.float32)
            y_[count] = np.array(read[-1], dtype=np.float32)
            count += 1

        # transform
        after_trans = transform(x_, y_, v_, trans_, (self.shape[3],self.shape[2]), augmentation=True)
        image = after_trans[:,:,0:3] - 128.0
        label = after_trans[:,:,3:]
        # get mask (no hard mining)
        mask = cls_sample(label)
        mask[mask>1]=1
        # scipy.io.savemat('debug.mat', dict(mask=mask, image=image, label=label))
        # cat image and mask
        image = image.transpose((2,0,1))
        label = label.transpose((2,0,1))
        image = np.append(image, mask[np.newaxis], axis=0)
        return image, label
