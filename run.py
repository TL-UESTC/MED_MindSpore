import numpy as np
import mindtorch.torch as torch
import mindtorch.torch.nn as nn
import mindtorch.torch.nn.functional as F
from mindtorch.torch.autograd import Function
from mindtorch.torch.utils.data import Dataset, DataLoader
import mindtorch.torch.optim as optim
import argparse
import mydata
from models import *
from train import *
from test import *
from pretrain import *
import itertools
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOC')
    parser.add_argument('--mkdir',type=str,default=None)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--source_temp',type=str,default='25')
    parser.add_argument('--target_temp',type=str,default='25')
    parser.add_argument('--test_set',type=str,default='Pan_test')
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--epochs',type=int,default=2000) 
    parser.add_argument('--eval_interval',type=int,default=200)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lamb1',type=float,default=0.1)
    parser.add_argument('--lamb2',type=float,default=0.1)
    parser.add_argument('--lamb3',type=float,default=1)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device id')
    # 
    parser.add_argument('--source_path',type=str,default=None)
    parser.add_argument('--target_path',type=str,default=None) 

    parser.add_argument('--ifsave',action='store_true')
    parser.add_argument('--load_model',action='store_true')
    parser.add_argument('--model_path',type=str,default='./saved_model/best.pt')
    parser.add_argument('--ifpretrain',action='store_true')
    args = parser.parse_args()
    
    models = {}
    models['conv'] = conv()
    models['lstm'] = lstm()
    models['fc1'] = fc1()
    models['fc2'] = fc2()
    

    criterion = {}
    criterion = nn.MSELoss(reduction='sum')
    optimizers = {}
    optimizers['conv'] = optim.Adam(models['conv'].parameters(),lr=args.lr)
    optimizers['lstm'] = optim.Adam(models['lstm'].parameters(),lr=args.lr)
    optimizers['fc1'] = optim.Adam(models['fc1'].parameters(),lr=args.lr)
    optimizers['fc2'] = optim.Adam(models['fc2'].parameters(),lr=args.lr)  
    optimizers['parameters'] = optim.Adam(itertools.chain(
    models['conv'].parameters(),
    models['lstm'].parameters(),
    models['fc1'].parameters(),
    models['fc2'].parameters()), lr=args.lr)
    
    
    source_data_path,source_train_set,source_test_set = get_trainpath(args.source_path)
    target_data_path,target_train_set,target_test_set = get_testpath(args.target_path)
    if args.mode == 'train':
        print('if save:{}'.format(args.ifsave))
        print('if pretrain:',args.ifpretrain)
        print('load model:{}'.format(args.load_model))
        print('epoch',args.epochs,'batch_size',args.batch_size)
        if args.load_model:
            print('model path:',args.model_path)
        if args.ifpretrain:
            # lh
            pretrain(args.mkdir,args.source_temp,args.target_temp,source_data_path,source_train_set,source_test_set,models, criterion, optimizers, args.batch_size,2000, 500, seed=100, device_type=args.device, ifsave=True,load_model=False)
        train(args.mkdir,args.source_temp,args.target_temp,source_data_path,source_train_set,source_test_set,target_data_path,target_train_set,target_test_set,models, criterion, optimizers, args.batch_size,args.epochs, args.eval_interval,args.lamb1,args.lamb2,args.lamb3, seed=100, device_type=args.device, ifsave=args.ifsave,load_model=args.load_model,model_path=args.model_path)
    elif args.mode == 'pretrain':
        args.ifsave = True
        models['fc1'].init_weight()
        models['fc2'].init_weight()
        print('if save:{}'.format(args.ifsave))
        print('load model:{}'.format(args.load_model))
        print('epoch',args.epochs,'batch_size',args.batch_size)
        pretrain(args.mkdir,args.source_temp,args.target_temp,source_data_path,source_train_set,source_test_set,models, criterion, optimizers, args.batch_size,args.epochs, args.eval_interval, seed=100, device_type=args.device, ifsave=args.ifsave,load_model=args.load_model)
    elif args.mode == 'test':
        print('test mode')
        print('test set:',args.test_set)
        data_path = None
        test_set = None
        if args.test_set == 'Pan_test':
            test_set = mydata.Pan_test_set
            data_path = mydata.Pan_data_path 
        elif args.test_set == 'Pan_train':
            test_set = mydata.Pan_train_set
            data_path = mydata.Pan_data_path 
       
        print('data path:',data_path)
        print('test set',test_set)
        test(args.mkdir,args.target_temp,models, data_path,test_set, seed=0, device_type=args.device,load_model_path=args.model_path)
  

