import os
import sys
import time
import numpy as np
import h5py
import pathlib
import pickle



def load_mat_files(f_name):
	acceptable_filenames = ("train_32x32.mat", "test_32x32.mat", "extra_32x32.mat",
	                        "train_digitStruct.mat", "test_digitStruct.mat", "train_32x32",
	                        "test_32x32", "extra_32x32", "train_digitStruct", "test_digitStruct")
	if f_name not in acceptable_filenames:
		print("Incorrect file name passed in")
		print(f"Filenames must be one of {acceptable_filenames}")
	if ".mat" not in f_name:
		f_name += ".mat"
	
	if "digitStruct" in f_name:
		import h5py
		import mat73
		import tables
		file = h5py.File(f_name, "r")
		# file = mat73.loadmat(f_name)
		print()
	else:
		from scipy.io import loadmat
		data = loadmat(f_name)
	print()

def read_name(f, index):
	# https://marcinbogdanski.github.io/ai-sketchpad/PyTorchNN/1350_PT_CNN_SVHN.html
    """Decode string from HDF5 file."""
    assert isinstance(f, h5py.File)
    assert index == int(index)
    ref = f['/digitStruct/name'][index][0]
    return ''.join(chr(v[0]) for v in f[ref])


def read_digits_raw(f, index):
	# https://marcinbogdanski.github.io/ai-sketchpad/PyTorchNN/1350_PT_CNN_SVHN.html
	"""Decode digits and bounding boxes from HDF5 file."""
	assert isinstance(f, h5py.File)
	assert index == int(index)
	
	ref = f['/digitStruct/bbox'][index].item()
	ddd = {}
	for key in ['label', 'left', 'top', 'width', 'height']:
		dset = f[ref][key]
		if len(dset) == 1:
			ddd[key] = [int(dset[0][0])]
		else:
			ddd[key] = []
			for i in range(len(dset)):
				ref2 = dset[i][0]
				ddd[key].append(int(f[ref2][0][0]))
	return ddd


def get_label(ddict):
	# https://marcinbogdanski.github.io/ai-sketchpad/PyTorchNN/1350_PT_CNN_SVHN.html
	"""Convert raw digit info into len-5 label and single bounding box"""
	assert isinstance(ddict, dict)
	
	# construct proper label for NN training
	# image '210' -> [3, 2, 1, 10, 0, 0]
	#                 ^  ^  ^  ^   ^--^-- "0, 0" pad with '0' (no digit)
	#                 |  ---------------- "210" house number, 0 encoded as 10
	#                 ------------------- "3" is number of digits
	label = ddict['label'].copy()
	label = [len(label)] + label + [0] * (5 - len(label))
	
	left = min(ddict['left'])
	top = min(ddict['top'])
	right = max(l + w for l, w in zip(ddict['left'], ddict['width']))
	bottom = max(t + h for t, h in zip(ddict['top'], ddict['height']))
	return tuple(label), (left, top, right, bottom)


def read_mat_file(filepath):
	# https://marcinbogdanski.github.io/ai-sketchpad/PyTorchNN/1350_PT_CNN_SVHN.html
	"""Open .mat file and read all the metadata."""
	assert isinstance(filepath, (str, pathlib.PosixPath))
	
	print(filepath)
	
	meta = {'names': [], 'labels': [], 'bboxes': []}
	with h5py.File(filepath) as f:
		length = len(f['/digitStruct/name'])
		for i in range(10): #length):
			name = read_name(f, i)
			ddict = read_digits_raw(f, i)
			label, bbox = get_label(ddict)
			meta['names'].append(name)
			meta['labels'].append(label)
			meta['bboxes'].append(bbox)
			if i % 1000 == 0 or i == length - 1:
				print(f'{i:6d} / {length}')
	return meta


def open_or_generate(name):
	# https://marcinbogdanski.github.io/ai-sketchpad/PyTorchNN/1350_PT_CNN_SVHN.html
	"""Either load .pkl, or if doesn't exit generate it and open."""
	assert name in ('extra', 'test', 'train')
	
	fname = name + '_digitStruct.mat' + '.pkl'
	if os.path.exists(os.getcwd() + "\\" + name + '_digitStruct.mat' + ".pkl"):
		with open(os.getcwd() + "\\" + name + '_digitStruct.mat' + ".pkl", 'rb') as f:
			meta = pickle.load(f)
			print(f'Loaded:{fname}')
	else:
		print(f'Generating {fname}:')
		meta = read_mat_file(os.getcwd() + "\\" + name + '_digitStruct.mat')
		with open(os.getcwd() + "\\" + name + '_digitStruct.mat' + ".pkl", 'wb') as f:
			pickle.dump(meta, f)
	
	return meta

if __name__ == "__main__":
	# open_or_generate("extra")
	with open("train_digitStruct.mat.pkl", "rb") as input_file:
		data = pickle.load(input_file)
	print()
