import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import sys


def main(sys):
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]
    
    net = torch.load('cnn.pkl') 
    test_output = net(test_x[:10]) 
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediciton number')
    print(test_y[:10].numpy(),'real number') 
    
    phototags = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

if __name__ == "__main__":
    main(sys.argv[1:])
