import mindtorch.torch as torch
import mindtorch.torch.nn as nn
from mindtorch.torch import sigmoid
from mindtorch.torch.nn.functional import relu
import numpy as np
import random
from collections import OrderedDict


filters = 256
kernel_size = 3
dropout = 0.2
hidden_units = 64
gain_fc1 = 0.25
gain_fc2 = 0.30

def init_seed(seed=0):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class conv(nn.Module):
  def __init__(self):
    super(conv, self).__init__()
    self.conv = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, filters, (3, 1), padding='same')),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(dropout)),
    ('conv2', nn.Conv2d(filters, filters, (3, 1), padding='same')),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(dropout))
]))

    
  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = x.reshape(-1,1,3,90)
    x = self.conv(x)
    x = x.reshape(-1,filters*3,90)
    return x

class lstm(nn.Module):
  def __init__(self):
    super(lstm, self).__init__()
    self.lstm1 = nn.Sequential(OrderedDict([
    ('lstm1', nn.LSTM(filters*3, hidden_units, 2, bidirectional=True, batch_first=True))
    ]))

    self.lstm2 = nn.Sequential(OrderedDict([
    ('lstm2', nn.LSTM(hidden_units*2, hidden_units, 2, bidirectional=True, batch_first=True)),
    ]))


  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = x.permute(0, 2, 1)
    x, (h,c) = self.lstm1(x)
    x, (h,c) = self.lstm2(x)
    return x

class fc1(nn.Module):
  def __init__(self):
    super(fc1, self).__init__()
    self.fc = nn.Sequential(OrderedDict([
    ('fc1linear1', nn.Linear(2*hidden_units, 2*hidden_units)),
    ('fc1relu', nn.ReLU()),
    ('fc1linear2', nn.Linear(2*hidden_units, 1))
]))


  def init_weight(self):
    nn.init.xavier_uniform_(self.fc[0].weight.data,gain=gain_fc1)
    nn.init.xavier_uniform_(self.fc[2].weight.data,gain=gain_fc1)

  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = self.fc(x) 
    return x

class fc2(nn.Module):
  def __init__(self):
    super(fc2, self).__init__()
    self.fc = nn.Sequential(OrderedDict([
    ('fc2linear1', nn.Linear(2*hidden_units, 2*hidden_units)),
    ('fc2relu', nn.ReLU()),
    ('fc2linear2', nn.Linear(2*hidden_units, 1))
]))


  def init_weight(self):
    nn.init.xavier_uniform_(self.fc[0].weight.data,gain=gain_fc2)
    nn.init.xavier_uniform_(self.fc[2].weight.data,gain=gain_fc2)

  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = self.fc(x) 
    return x


def discrepancy(out1, out2):
    return torch.sum(torch.abs(out1- out2))

def save_model(models, optimizers, loss_min, seed, model_path='./saved_model/best.pt'):
  torch.save({
        'conv':models['conv'].state_dict(),
        'lstm':models['lstm'].state_dict(),
        'fc1':models['fc1'].state_dict(),
        'fc2':models['fc2'].state_dict(),
        # 'optimizer_conv':optimizers['conv'].state_dict(),
        # 'optimizer_lstm':optimizers['lstm'].state_dict(),
        # 'optimizer_fc1':optimizers['fc1'].state_dict(),
        # 'optimizer_fc2':optimizers['fc2'].state_dict(),
        'optimizer_parameters':optimizers['parameters'].state_dict(),
        'loss_min':loss_min,
        'seed':seed
        }, model_path )

def load_saved_model(device,models,optimizers,loss_min,seed,model_path='./saved_model/best.pt'):
    device = torch.device('cuda')
    ckpt = torch.load(model_path,map_location=device)
    layers=['conv','lstm','fc1','fc2']
    for l in layers:
        models[l].load_state_dict(ckpt[l])
        #optimizers[l].load_state_dict(ckpt['optimizer_'+l])
    loss_min = ckpt['loss_min']
    seed = ckpt['seed']

