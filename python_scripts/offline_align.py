import os
import cv2
from skimage import transform as trans
import numpy as np
import math
import scipy.io

x_0 = [41.9891, 84.7495, 63.7709, 45.8549, 83.6355]
y_0 = [40.9533, 39.5924, 65.5459, 87.4031, 86.5053]
src = np.array( (x_0, y_0) ).transpose([1,0]).astype(np.float32)

def transform(x, y, v, img, imgSize, augmentation=False):
	src_r = src[np.nonzero(np.array(v)>0.5)]
	img2 = img.copy()
	dst = np.array( (x, y) ).transpose([1,0]).astype(np.float32)
	dst_r = dst[np.nonzero(np.array(v)>0.5)]
	tform = trans.SimilarityTransform()
	tform.estimate(dst_r, src_r)
	M = tform.params[0:2,:]
	# print(M)
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
	return out

def main():
	root_dir = '/Data/'
	save_dir = os.path.join(root_dir, 'helen_align_128')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	split_im = os.path.join(root_dir, 'helen/train_dgx','test_im.txt')
	img_list = open(split_im, 'r').read().splitlines()
	split_lb = os.path.join(root_dir, 'helen/train_dgx','test_lb.txt')
	lab_list = open(split_lb, 'r').read().splitlines()

	for id in range(len(img_list)):	
		im_name = os.path.join(root_dir, img_list[id])
		print(img_list[id])
		img = cv2.imread(im_name)
		lb_root = os.path.join(root_dir, lab_list[id])
		name_ = lb_root.rpartition('/')
		name_ = name_[-1]
		if os.path.exists(os.path.join(save_dir, '{}.mat'.format(name_))):
			continue
		# get lb list
		for lb in range(11):

			lb_name = os.path.join(lb_root, name_+"_lbl{:02}.png".format(lb))
			lab_ = np.array(cv2.imread(lb_name, cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0
			if lb == 0:
				lab = lab_[:,:,np.newaxis]
			else:
				lab = np.append(lab, lab_[:,:,np.newaxis], axis=2)
				
		trans_ = np.append(img, lab, axis=2)
		# print(trans_.shape)
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
		# print(x_)
		out = transform(x_, y_, v_, trans_, (128,128), augmentation=False)
		# print(out.shape)
		scipy.io.savemat(os.path.join(save_dir, '{}.mat'.format(name_)), dict(out=out))

if __name__ == "__main__":
	main()