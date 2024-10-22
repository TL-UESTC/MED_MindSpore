from mindtorch.torch.utils.data import Dataset, DataLoader
import mindtorch.torch.optim as optim
from mindtorch.torch.autograd import Variable
import mindspore as ms
import argparse
from tqdm import tqdm
import math
from utils import *  
from mydata import *
from models import *
from mindspore import Tensor
eval_interval = 100
batch_size = 6

# lh
def pretrain(rundir,source_temp,target_temp,source_data_path,source_train_set,source_test_set,models, criterion, optimizers, batch_size, epochs,eval_interval, seed=0, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'), ifsave=True, load_model=False, load_model_path='./saved_model/best.pt'):
  loss_min = 10000
  rundir = mkdir(rundir)
  device = torch.device(device_type)
  if load_model:
      load_saved_model(device,models,optimizers,loss_min,seed)
  if torch.cuda.is_available():
    for  model in models:
      models[model].to(device)
  
  init_seed(seed)
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()
  for temp_idx in range(1):
    source_data = Mydataset(source_data_path, source_temp, source_train_set,mode='train')
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    source_test_data = Mydataset(source_data_path, source_temp, source_test_set,mode='test')
    source_test_loader = DataLoader(source_test_data, batch_size=1, shuffle=False)
    #target_test_data = Mydataset(target_data_path, target_temp, target_test_set, mode='test')
    #target_test_loader = DataLoader(target_test_data, batch_size=1, shuffle=False)
    
    loss_iter_train = []
    loss_iter_predictor = []
    loss_iter_test = []
    loss_iter_mae = []
    loss_iter_rmse = []
    loss_iter_max = []
    loss_iter_domain_acc = []
    loss_train_predictor = 0
    test_len = len(source_test_loader)
    min_max,min_mae,min_rmse = [],[],[]
    for i in range(test_len):
      min_mae.append(1)
      min_rmse.append(1)
      min_max.append(1)
    #checkpoint = torch.load(load_model_path, map_location=device)
     #models['domain_classifier'].load_state_dict(checkpoint['domain_classifier'])
    
    
    
    # for epoch in range(epochs+500):
    #   ##########
    #   #train
    #   ##########
    #   for model in models:
    #     models[model].train()
    #   loss_train = 0
    #   loss_test = 0
    #   source_sample = 0
    #   #tqdm_mix = tqdm(source_loader,desc='epoch '+str(epoch))
    #   for i, (source_data, source_label) in enumerate(source_loader):
    #     source_data = source_data.to(device)
    #     source_label = source_label.to(device)

    #     # for op in optimizers:  ??
    #     #   optimizers[op].zero_grad()

    #     predict_p1 = models['fc1'](models['lstm'](models['conv'](source_data))).squeeze()
    #     predict_p2 = models['fc2'](models['lstm'](models['conv'](source_data))).squeeze()
    #     predict_loss = criterion(predict_p1,source_label) + criterion(predict_p2,source_label)
    #     # predict_loss.backward() ??
    #     for op in optimizers:
    #       optimizers[op].step()
    #     loss_train += predict_loss.item()
    #     source_sample += len(source_data)
    #     #if ((epoch+1) % eval_interval) == 0:
    #     #  plot_result(source_label, predict_label, save_image='train', 
    #     #          test_name=source_data_path[7:-1] + source_temp + '_epoch_' + str(epoch))
    #   loss_train = loss_train/(source_sample)
    #   print('epoch {}:loss {}'.format(epoch, loss_train))
    #   if (loss_train < loss_min) & (ifsave==True):
    #     loss_min = loss_train
    #     path = rundir+'/saved_model/best.pt'
    #     #save_model(models, optimizers, loss_min, seed,path)
    #     #print('min loss:{} saved model'.format(loss_min))
      


    # 定义网络

    # 定义前向过程
    def forward_fn(data, label):
      predict_p1 = models['fc1'](models['lstm'](models['conv'](data))).squeeze()
      predict_p2 = models['fc2'](models['lstm'](models['conv'](data))).squeeze()
      loss = criterion(predict_p1, label) + criterion(predict_p2, label)
      return loss, predict_p1, predict_p2

    # 反向梯度定义
    # parameters = [optimizer.parameters() for optimizer in optimizers.values()]
    # parameters=optimizers['conv'].parameters+optimizers['lstm'].parameters+ optimizers['fc1'].parameters+ optimizers['fc2'].parameters
    grad_fn = ms.value_and_grad(forward_fn, None,optimizers['parameters'].parameters, has_aux=True)

    # optimizer=[optim for optim in optimizers]
    # 单步训练定义
    def train_step(data, label):
        (loss, _, _), grads = grad_fn(data,label)
        # for op in optimizers: 
        #     optimizers[op](grads)
        optimizers['parameters'](grads)
        
        # opp=optimizers['fc2'].parameters
        # print(opp)
        return loss

    # 数据迭代训练
    for epoch in range(epochs+500):
      for model in models:
        models[model].train()      
      loss_train = 0
      loss_test = 0
      source_sample = 0
      for i, (source_data, source_label) in enumerate(source_loader):
          source_data = source_data.to(device)
          source_label = source_label.to(device)

# ！！
          loss = train_step(source_data, source_label)
          
          loss_train += loss.asnumpy().item()
          source_sample += len(source_data)
      loss_train = loss_train / source_sample
      print('epoch {}:loss {}'.format(epoch, loss_train))
      if (loss_train < loss_min) & (ifsave==True):
          loss_min = loss_train
          path = rundir+'/saved_model/best.pt'


      ##########
      #test
      ##########
      for model in models:
        models[model].eval()
      
      #tqdm_test = tqdm(source_test_loader, desc='source data test')
      loss_mae = 0
      loss_rmse = 0
      loss_max = 0
      if ((epoch+1) % eval_interval) == 0:
            print('source test res')
      for i, data in enumerate(source_test_loader):
        x_test, y_test = data
        x_test, y_test = x_test.squeeze(), y_test.squeeze()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
          predict_p1 = models['fc1'](models['lstm'](models['conv'](x_test))).squeeze()
          predict_p2 = models['fc2'](models['lstm'](models['conv'](x_test))).squeeze()
        y_predict = np.concatenate([predict_p1.cpu(),predict_p2.cpu()])
        y_test = np.concatenate([y_test.cpu(),y_test.cpu()])
        #print(type(y_predict),type(y_test))
        y_predict = torch.from_numpy(y_predict.astype(np.float32)).cuda()
        y_test = torch.from_numpy(y_test.astype(np.float32)).cuda()
        loss_mse = criterion_mse(y_predict, y_test)
        loss_test = loss_mse.detach().cpu().item()
        loss_mae = criterion_mae(y_predict, y_test).detach().cpu().item()
        loss_rmse = math.sqrt(loss_mse.detach().cpu().item())
        loss_max = MAXLoss(y_predict, y_test)
        if epoch > epochs:  
          min_avg = (min_mae[i] + min_rmse[i])/2
          loss_avg = (loss_mae + loss_rmse) / 2
          if min_avg > loss_avg:
            min_max[i] = loss_max
            min_rmse[i] = loss_rmse
            min_mae[i] = loss_mae
            test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]+'best'
            
            plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          
            error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
            print(error)
            save_error(rundir,error,test_name,'test')
            loss_min = loss_train
            path = rundir+'/saved_model/best.pt'
            save_model(models, optimizers, loss_min, seed,path)
            print('min avg loss:{} saved model'.format(loss_avg))

        save_min(rundir,min_mae,min_rmse,min_max,epoch)
        if ((epoch+1) % eval_interval) == 0:
          test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')
      #####
      loss_iter_train.append(loss_train)
      loss_iter_predictor.append(loss_train_predictor)
      loss_iter_test.append(loss_test)
      loss_iter_mae.append(loss_mae)
      loss_iter_rmse.append(loss_rmse)
      loss_iter_max.append(loss_max)
      #if ( ((epoch+1) % eval_interval)==0 ) & (ifsave==True):
      #  save_model(models, optimizers, loss_min, seed, model_path='./saved_model/epoch'+str(epoch)+'.pt')
    plot_train_loss(rundir,loss_iter_train, loss_iter_train, loss_iter_train, loss_iter_train, epochs)
    plot_test_loss(rundir,loss_iter_mae, loss_iter_rmse, loss_iter_max, epochs)
    
