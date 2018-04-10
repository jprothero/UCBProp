import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ARGS:
    def __init__(self):
        self.batch_size=128
        self.test_batch_size=1000
        self.epochs = 10
        self.lr=0.01
        self.momentum=0.5
        self.no_cuda=False
        self.seed = 1
        self.log_interval=10
args = ARGS()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
from ipdb import set_trace
def ucb_train(az):
    az.model.eval()    
    for batch_idx, (_data, _target) in enumerate(train_loader):
        for _ in range(az.cycles_per_batch):
            batch_total = _target.size()[0]
            if args.cuda:
                _data, _target = _data.cuda(), _target.cuda()
            data, target = Variable(_data), Variable(_target)
            for s in range(az.num_sims):
                print("Sim {} of {}".format(s, az.num_sims))
                for i in range(az.num_steps):
                    az.update_params_step(az.step_nodes[i])                    
                    output = az.model(data)
                    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                    batch_correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    batch_accuracy = batch_correct/batch_total
                az.backup_step(batch_accuracy)
            for step_node in az.step_nodes:
                az.update_params_step(step_node, visits=True)
            output = az.model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
            batch_accuracy = batch_correct/batch_total
            print('Batch: {}, Batch Accuracy: {:.3f}%'.format(batch_idx, batch_accuracy))
            az.reset_az()
            
            # az.step(batch_accuracy, update_params=True)
            

                # if batch_idx == 9:
                #     print(list(az.model.parameters())[0])
                # if batch_idx == 10:
                #     print(list(az.model.parameters())[0])                
                #     return

def ucb_test(az):
    correct = 0
    total = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        total += target.size()[0]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    accuracy = correct/total
    print('\nTest set: Accuracy: {:.0f}%\n'.format(100. * accuracy))


# #normal
# for epoch in range(1, args.epochs + 1):
#     train(epoch)
#     test()

#UCB prop
# from async_alphazero import AsyncAlphazero

# az = AsyncAlphazero(model=model, 
# num_slices=4, 
# c=10, 
# cycles_per_batch=100, 
# num_sims=40,
# num_steps=5,
# lr=1)

#would be good if we had a lstm or something which produced a prior, and it was
#improved by improved search probas

#so.... the issue is that they are sinking a lot because of the UCT always picking the same
#that will severely limit the capacity of the net 

# ucb_train(az)
# ucb_test(az)

from ipdb import set_trace
from models import MetaLearner

meta_net = MetaLearner(model, (1, 28, 28), model_optim=optimizer)

model.train()
meta_net.train()
for batch_idx, (data, target) in enumerate(train_loader):
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)

    meta_net(data, target)

# for batch_idx, (data, target) in enumerate(train_loader):
#     if args.cuda:
#         data, target = data.cuda(), target.cuda()
    
#     data, target = Variable(data), Variable(target)
#     # optimizer.zero_grad()
#     meta_net(data, target, train=True)
#     print('Batch: {}, Accuracy: {:.3f}'.format(batch_idx, batch_accuracy))