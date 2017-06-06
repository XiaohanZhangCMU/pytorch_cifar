import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
#from utils import progress_bar
import cnn_models

'''
To do: 
    write a dataloader for arbitrary data
    use cnn for dislocation detection problem
    write grid search to find optimal LR and batch size
    separate MNIST and CIFAR10. How to organize codes?
    utils.progressbar problem when submit jobs
'''


EPOCH = 5
BATCH_SIZE = to_be_replaced #150
LR = to_be_replaced/10000.  #0.0022
DOWNLOAD_MNIST = False
DOWNLOAD_CIFAR10 = False
nnfile = 'cnn.pkl' 
nnparamfile = 'cnn.pkl.params'
use_cuda = torch.cuda.is_available()

def train_and_save( net, train_loader, validation_x, validation_y, lr, EPOCH, nnfile, nnparamfile):
    #SGD: %12. #ADM: 43%.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    train_loss = 0 
    total = 0
    correct = 0
    for EPOCH in range(EPOCH):
        for batch_idx, (x,y) in enumerate(train_loader):
            if (use_cuda): 
                x, y = x.cuda(), y.cuda()
            b_x, b_y = Variable(x), Variable(y)
            prediction = net(b_x)
            loss = loss_function(prediction, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch_idx+1) % 50 == 0: 
#                test_output = net(validation_x)
#                pred_y = torch.max(test_output,1)[1].data.squeeze()
#                accuracy = sum(pred_y == validation_y) / validation_y.size(0)
#                print ('Epoch: ', EPOCH, '| train loss: %.4f' % loss.data[0], '| test accuracy', accuracy)
#
                train_loss += loss.data[0]
                _, predicted = torch.max(prediction.data,1)
                total += b_y.size(0)
                correct += predicted.eq(b_y.data).cpu().sum()
                if (0) : # progress bar gives runtime error if running on nodes. Interactively fun.
                    progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %3f%% (%d%d)'
                        %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
    torch.save(net, nnfile+'.'+str(LR)+'.'+str(BATCH_SIZE))
    torch.save(net.state_dict(), nnparamfile+'.'+str(LR)+'.'+str(BATCH_SIZE))
    f = open('Acc_'+str(LR) +'_'+ str(BATCH_SIZE)+'.txt','w')
    f.write(str(100.*correct/total))
    return net

# train to recognize MNIST data
def main_1(argv):
    train_data = torchvision.datasets.MNIST(
        root = '../mnist',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE,
                                   shuffle = True, num_workers = 2)
    test_data = torchvision.datasets.MNIST(root='../mnist/', train=False)
    validation_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
    validation_y = test_data.test_labels[:2000]

    net = cnn_models.CNN( )
    net = train_and_save( net, train_loader, validation_x, validation_y, LR, EPOCH, nnfile, nnparamfile)

# train to reconginize CIFAR10 data
def main(argv):
    train_data = torchvision.datasets.CIFAR10(
        root = '../cifar10',
        train = True,
        transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        download=DOWNLOAD_CIFAR10
    )
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE,
                                   shuffle = True, num_workers = 2)
    test_data = torchvision.datasets.CIFAR10(root='../cifar10/', train=False)
    validation_x = Variable(torch.unsqueeze(torch.from_numpy(test_data.test_data), dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
    validation_y = test_data.test_labels[:2000]

    net = cnn_models.CNN( )
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
        torch.backends.cudnn.enabled=True
        
    net = train_and_save( net, train_loader, validation_x, validation_y, LR, EPOCH, nnfile, nnparamfile)

if __name__ == "__main__":
    main(sys.argv[1:])
