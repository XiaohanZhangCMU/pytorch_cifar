import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import sys
from utils import progress_bar
import torch
import cifar10

use_cuda = torch.cuda.is_available()

def main1(sys):
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]
    
    net = torch.load('cnn.pkl') 
    test_output = net(test_x[:10]) 
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediciton number')
    print(test_y[:10].numpy(),'real number') 
    
def main(sys):
    phototags = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    test_data = cifar10.CIFAR10(root='../../cifar10/', train=False, download=False)
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 100, shuffle=True, num_workers=2) 

#    net = torch.load('cnn.pkl.0.0028.128') 
    net = torch.load('cnn.pkl.0.1.128')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_function = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
                                                                                            
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
     
if __name__ == "__main__":
    main(sys.argv[1:])
