import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import cnn_models

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
DOWNLOAD_CIFAR10 = True
nnfile = 'cnn.pkl' 
nnparamfile = 'cnn.pkl.params'

def train_and_save( net, train_loader, validation_x, validation_y, lr, EPOCH, nnfile, nnparamfile):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    for EPOCH in range(EPOCH):
        for step, (x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            prediction = net(b_x)
            loss = loss_function(prediction, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0: 
                test_output = net(validation_x)
                pred_y = torch.max(test_output,1)[1].data.squeeze()
                accuracy = sum(pred_y == validation_y) / validation_y.size(0)
                print ('Epoch: ', EPOCH, '| train loss: %.4f' % loss.data[0], '| test accuracy', accuracy)

    torch.save(net, nnfile)
    torch.save(net.state_dict(), nnparamfile)
    return net


def main_1(argv):
    train_data = torchvision.datasets.MNIST(
        root = './mnist',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE,
                                   shuffle = True, num_workers = 2)
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    validation_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
    validation_y = test_data.test_labels[:2000]

    net = cnn_models.CNN( )
    net = train_and_save( net, train_loader, validation_x, validation_y, LR, EPOCH, nnfile, nnparamfile)

def main(argv):

    train_data = torchvision.datasets.CIFAR10(
        root = './cifar10',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download=DOWNLOAD_CIFAR10
    )
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE,
                                   shuffle = True, num_workers = 2)
    test_data = torchvision.datasets.CIFAR10(root='./cifar10/', train=False)
    validation_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
    validation_y = test_data.test_labels[:2000]

    net = cnn_models.CNN( )
    net = train_and_save( net, train_loader, validation_x, validation_y, LR, EPOCH, nnfile, nnparamfile)

if __name__ == "__main__":
    main(sys.argv[1:])
