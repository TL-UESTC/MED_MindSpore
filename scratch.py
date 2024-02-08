print(1)
import mindspore as ms
net = LeNet().to(config_args.device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

def forward_fn(X,y):
    out=net(X)
    loss=criterion(out,y)
    return loss,out

grad_fn=ms.ops.value_and_grad(forward_fn,None,optimizer.parameters,has_aux=True)

def train_step(X,y):
    (loss,_),grads=grad_fn(X,y)
    optimizer(grads)
    return loss

net.train()

# 数据迭代训练
for i in range(epochs):
    for X, y in train_data:
        X, y = X.to(config_args.device), y.to(config_args.device)
        res=train_step(X,y)
        print("------>epoch:{}, loss:{:.6f}".format(i, res))


# MSAdapter 模型训练
def train(config_args):
    train_images = datasets.CIFAR10('./', train=True, download=True, transform=transform)
    train_data = DataLoader(train_images, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
 
    epochs = config_args.epoch
    net = AlexNet().to(config_args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = ms.nn.SGD(net.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=0.0005)
    loss_net = ms.nn.WithLossCell(net, criterion)
    train_net = ms.nn.TrainOneStepCell(loss_net, optimizer)
    net.train()
    print("begin training ......")
    for i in range(epochs):
        for X, y in train_data:
            res = train_net(X, y)
            print("---------------------->epoch:{}, loss:{:.6f}".format(i, res.asnumpy()))
    torch.save(net.state_dict(), config_args.save_path)