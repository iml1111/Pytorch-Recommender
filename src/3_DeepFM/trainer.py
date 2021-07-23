from copy import deepcopy
import torch
from tqdm import trange, tqdm
import numpy as np

class Trainer:
	
	def __init__(self, model, optim, crit, device):
		self.model = model
		self.optim = optim
		self.crit = crit
		self.device

	def _train(self, data_loader, config):
		self.model.train()
		total_lost = 0
		