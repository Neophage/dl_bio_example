import argparse
import json
import os
from os.path import isdir, join
import numpy as np

import torch
from DLBio.pytorch_helpers import get_device

import user_config as config

from datasets.data_getter import get_test_dataloader, get_class_names
from models.model_getter import get_model

def get_options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_path', type=str, default='./experiments/_debug/model_resnet18_imfx_im.pt')
	parser.add_argument('--out_path', type=str, default='./experiments/_debug/conf_mat.png')
	parser.add_argument('--dataset', type=str, default='imfx_im')
	parser.add_argument('--subset', type=str, default='positive')
	parser.add_argument('--split_index', type=int, default=0)
	parser.add_argument('--device', type=int, default=None)
	parser.add_argument('--in_dim', type=int, default=3)
	parser.add_argument('--out_dim', type=int, default=7)
	parser.add_argument('--model_type', type=str, default=config.MT)

	return parser.parse_args()
	
def run(options):
	if options.device is not None:
		pt_training.set_device(options.device)

	device = get_device()
	
	# load model
	model = get_model(
		options.model_type,
		options.in_dim,
		options.out_dim,
		device
	)

	if options.model_path is not None:
		model_sd = torch.load(options.model_path)
		model.load_state_dict(model_sd, strict=False)
		
	model.eval()
	
	data_loader = get_test_dataloader(options.dataset, options.subset, options.split_index)
	class_names = get_class_names(options.dataset)

	conf_mat = _generate_conf_mat(options, model, data_loader, device)
	np.set_printoptions(suppress=True)
	print(conf_mat)
	
	f1_scores = calculate_f1_scores(options, conf_mat)
	print('\nF1 Scores:')
	for label in range(options.out_dim):
		print(class_names[label] + '\t{:.3f}'.format(f1_scores[label]))
	
# generate a confusion matrix from a given model and data loader
def _generate_conf_mat(options, model, data_loader, device):
	conf_mat = np.zeros([options.out_dim, options.out_dim])
	
	for sample in data_loader:
		image, label = sample
		label = label.item()
		x = image.to(device)
		out = model(x)
		pred = out.argmax(1).item()
		conf_mat[label, pred] += 1
		
	return conf_mat
	
# calculate classwise f1 scores from a confusion matrix
def calculate_f1_scores(options, conf_mat):
	f1_scores = np.zeros(options.out_dim)
	
	for label in range(options.out_dim):
		tp = conf_mat[label, label]
		fp = np.sum(np.delete(conf_mat[:, label], label))
		fn = np.sum(np.delete(conf_mat[label, :], label))
		
		f1_scores[label] = tp / (tp + 0.5 * (fp + fn))
		
	return f1_scores
	

if __name__ == "__main__":
	OPTIONS = get_options()
	run(OPTIONS)