import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ARGS:
    def __init__(self):
        self.batch_size=64
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
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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

def ucb_train(az):
    for batch_idx, (_data, _target) in enumerate(train_loader):
        for _ in range(az.cycles_per_batch):
            batch_total = _target.size()[0]
            if args.cuda:
                _data, _target = _data.cuda(), _target.cuda()
            data, target = Variable(_data), Variable(_target)
            for _ in range(az.num_sims):
                az.model.eval()
                output = az.model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                batch_correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
                batch_accuracy = batch_correct/batch_total
                

                az.step(batch_accuracy)

            # az.step(batch_accuracy, update_params=True)
            print('Batch: {}, Batch Accuracy: {:.3f}%'.format(batch_idx, batch_accuracy))
            

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
from async_alphazero import AsyncAlphazero

az = AsyncAlphazero(model=model, num_slices=5, c=3, cycles_per_batch=100, num_sims=10, reset_every=15)

#would be good if we had a lstm or something which produced a prior, and it was
#improved by improved search probas

#so.... the issue is that they are sinking a lot because of the UCT always picking the same
#that will severely limit the capacity of the net 

ucb_train(az)
# ucb_test(az)