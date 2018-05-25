import caffe
import os
import cv2
from skimage import transform as trans
import numpy as np
import math
import scipy.io
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='test face parsing')
	parser.add_argument('--root_dir', type=str, default = "/Data/")
	parser.add_argument('--model', type=str, default = 'models/state_iter_20000.caffemodel')
	parser.add_argument('--device_id', type=int, default = 2)
	parser.add_argument('--proto', type=str, default = 'models/deploy.prototxt')

	args = parser.parse_args()
	return args

def fast_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def test_seg(trans_, net, layer='ae_prob'):
	image = trans_[:,:,0:3] - 128.0
	label = trans_[:,:,3:-1]
	image = image.transpose((2,0,1))
	net.blobs['data'].reshape(1, *image.shape)
	net.blobs['data'].data[...] = image
	net.forward()
	prob = net.blobs[layer].data[0]
	prob_hard = prob[:-1].argmax(0)
	label_hard = label.transpose((2,0,1)).argmax(0)
	hist_score = fast_hist(label_hard.flatten(),
						prob_hard.flatten(),
						10)
	return hist_score

def main(args):
	caffe.set_device(args.device_id)
	caffe.set_mode_gpu()
	split_im = os.path.join(args.root_dir, 'helen/train_dgx','test_im.txt')
	img_list = open(split_im, 'r').read().splitlines()
	split_lb = os.path.join(args.root_dir, 'helen/train_dgx','test_lb.txt')
	lab_list = open(split_lb, 'r').read().splitlines()
	net = caffe.Net(args.proto, args.model, caffe.TEST)
	hist = np.zeros((10, 10))
	
	for id in range(len(img_list)):
		lb_name = lab_list[id]
		print(lb_name)
		name_ = lb_name.rpartition('/')
		name_ = name_[-1]
		trans_name = os.path.join(args.root_dir, 'helen_align_128/{}.mat'.format(name_))
		trans_ = scipy.io.loadmat(trans_name)
		trans_ = trans_['out']
		hist += test_seg(trans_, net, layer='ae_prob')

	iu = (np.diag(hist) + 0.001) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 0.001)
	print('>>>', 'models', args.model, 'IU mean', iu[1])
	print('>>>', 'models', args.model, 'IU', iu)

	return hist

if __name__ == "__main__":
	args = parse_args()
	main(args)