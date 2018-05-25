import caffe
import numpy as np
import scipy.io
import math
import os
import sys
import refine_util as rv
"""
python train.py 2 0 ae_label
"""

if __name__ == "__main__":
	caffe.set_mode_gpu()
	fid = sys.argv[1]
	device_id = sys.argv[2]
	caffe.set_device(int(device_id))
	solverproto = 'models/solver_v{}.prototxt'.format(fid)
	Sov = rv.parse_solverproto(solverproto)
	save_path = Sov['snapshot_prefix'].rpartition('/')
	save_path = save_path[0]
	print(save_path)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	solver = caffe.SGDSolver(solverproto)
	solver.set_iter(0)
	max_iter = 20001;
	save_iter = 10;
	display_iter = 10
	_train_loss = 0
	tmpname = os.path.join(save_path, 'loss.mat')
	cur_res_mat = os.path.join(save_path, 'infer_res.mat')
	cur_iter = os.path.join(save_path, 'iter.mat')
	train_loss = np.zeros(int(math.ceil(max_iter/ display_iter)))

	if len(sys.argv) > 4:
		solver.net.copy_from(os.path.join('models', sys.argv[4]))

	if not os.path.exists(cur_iter):
		solver.step(1)
		solver.set_iter(1)
		batch, label, active = rv.getbatch(solver.net, layer=sys.argv[3])
		scipy.io.savemat(cur_res_mat, dict(batch = batch, label = label, active = active))
	else:
		curiter = scipy.io.loadmat(cur_iter)
		curiter = curiter['cur_iter']
		curiter = int(curiter)
		solver.set_iter(curiter)
		train_loss_load = scipy.io.loadmat(tmpname)
		train_loss_load = np.array(train_loss_load['train_loss'], dtype=np.float32).squeeze()
		train_loss[:len(train_loss_load)]=train_loss_load
		solverstate = Sov['snapshot_prefix'] + \
			'_iter_{}.solverstate'.format(solver.iter)
		caffemodel = Sov['snapshot_prefix'] + \
			'_iter_{}.caffemodel'.format(solver.iter)
		if os.path.exists(solverstate):
			solver.restore(solverstate)
		elif os.path.exists(caffemodel):
			solver.net.copy_from(caffemodel)
		else:
			raise Exception("Model does not exist.")

	begin = solver.iter
	_train_loss = 0

	for iter in range(begin, max_iter):
		solver.step(1)
		_train_loss += solver.net.blobs['loss'].data
		if iter % display_iter == 0:
			train_loss[int(iter / display_iter)] = _train_loss / display_iter
			_train_loss = 0
		if iter % save_iter == 0:
			batch, label, active = rv.getbatch(solver.net, layer=sys.argv[3])
			scipy.io.savemat(cur_res_mat, dict(batch = batch, label = label, active = active))
			scipy.io.savemat(cur_iter, dict(cur_iter = iter))
			scipy.io.savemat(tmpname, dict(train_loss = train_loss))
		if (iter-1) % Sov['snapshot'] == 0:
			rv.clear_history(Sov['snapshot'],Sov['snapshot_prefix'],iter)
