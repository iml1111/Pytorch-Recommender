import torch.nn as nn


class AutoEncoder(nn.Module):

	def __init__(self, n_input, act_func):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_input, 200, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(200),
			
			nn.Linear(200, 100, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(100),

			nn.Linear(100, 50, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(50),

			nn.Linear(50, 25, bias=True),
		)
		self.decoder = nn.Sequential(
			nn.Linear(25, 50, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(50),
			
			nn.Linear(50, 100, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(100),
			
			nn.Linear(100, 200, bias=True),
			self.activation(act_func),
			nn.BatchNorm1d(200),
			
			nn.Linear(200, n_input, bias=True),
		)

	def forward(self, x):
		"""
		x = (user_nums,)
		- 해당 아이템에 대한 각 사용자의 평점 sparse matrix
		"""
		# Re-feeding???
		return self.decoder(self.encoder(x))

	@staticmethod
	def activation(kind):
	    if kind == 'selu':
	      return nn.SELU()
	    elif kind == 'relu':
	      return nn.ReLU()
	    elif kind == 'relu6':
	      return nn.ReLU6()
	    elif kind == 'sigmoid':
	      return nn.Sigmoid()
	    elif kind == 'tanh':
	      return nn.Tanh()
	    elif kind == 'elu':
	      return nn.ELU()
	    elif kind == 'lrelu':
	      return nn.LeakyReLU()
	    else:
	      raise ValueError('Unknown non-linearity type')