from ipdb import set_trace

import os, sys

sys.path.append(os.getcwd())

import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
def az(batch_of_fakes, D, num_sims=10, num_slices=100, c=1):
    root_node = {
        "children": None, "parent": None, "N": 0
    }

    linspace = np.linspace(0, 1, num_slices)

    improved_fakes = []

    batch_idx = 0
    for _ in tqdm(range(batch_of_fakes.shape[0])):
        fake = batch_of_fakes[batch_idx].clone()
        new_fake = batch_of_fakes[batch_idx].clone()
        batch_idx += 1
        real_idx = 0    
        while real_idx < len(fake):
            for _ in range(num_sims):
                fake_copy = fake.clone()
                curr_node = root_node
                exit_early = False
                prior_idx = 0
                while curr_node["children"] is not None:
                    if prior_idx < (len(fake_copy)-1):
                        max_uct = -1e7
                        max_uct_idx = None
                        for i, child in enumerate(curr_node["children"]):
                            child["U"] = (
                                c*child["P"]*np.log(curr_node["N"]))/(1 + child["N"])
                            if (child["Q"] + child["U"]) > max_uct:
                                max_uct_idx = i
                        curr_node = curr_node["children"][max_uct_idx]
                        fake_copy[prior_idx] = linspace[max_uct_idx]
                        prior_idx += 1
                    else:
                        exit_early = True
                        break

                if not exit_early:
                    prior = fake_copy[prior_idx].detach().data.numpy()
                    value = (D(fake_copy).squeeze()/2) + .5
                    linspace_prior = -np.log(np.abs(linspace - prior)+1e-7)
                    linspace_prior /= linspace_prior.sum()

                    curr_node["children"] = []

                    for p in linspace_prior:
                        U = p

                        child = {
                            "N": 0,
                            "W": 0,
                            "Q": 0,
                            "U": U,
                            "P": p,
                            "children": None,
                            "parent": curr_node
                        }

                        curr_node["children"].append(child)

                    while curr_node["parent"] is not None:
                        curr_node["N"] += 1
                        curr_node["W"] += value
                        curr_node["Q"] = curr_node["W"]/curr_node["N"]

                        curr_node = curr_node["parent"]

                    root_node = curr_node
                    root_node["N"] += 1
                else:
                    while curr_node["parent"] is not None:
                        curr_node = curr_node["parent"]

                    root_node = curr_node

            visits_idx = np.argmax([child["N"]
                                for child in root_node["children"]])

            new_fake[real_idx] = linspace[visits_idx]
            root_node = root_node["children"][visits_idx]
            root_node["parent"] = None
            real_idx += 1

        improved_fakes.append(new_fake.unsqueeze(0))
    
    improved_fakes = torch.cat(improved_fakes)
    
    return improved_fakes


torch.manual_seed(1)


MODE = 'wgan-gp'  # wgan or wgan-gp
DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for
use_cuda = False

# ==================Definition Start======================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True)
    if use_cuda:
        points_v = points_v.cuda()
    disc_map = netD(points_v).cpu().data.numpy()

    noise = torch.randn(BATCH_SIZE, 2)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise, volatile=True)
    true_dist_v = autograd.Variable(torch.Tensor(true_dist).cuda() if use_cuda else torch.Tensor(true_dist))
    samples = netG(noisev, true_dist_v).cpu().data.numpy()

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    if not FIXED_GENERATOR:
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    # plt.savefig('tmp/' + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')

    frame_index[0] += 1


# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) / BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print (netG)
print (netD)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen()

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev, real_data_v).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    if not FIXED_GENERATOR:
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        _data = next(data)
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()

        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)
        
        min = fake.min()
        max = fake.max()
        fake = (fake - min)/(max - min)
        improved_fakes = az(fake, netD)
        improved_fakes = (improved_fakes + min)*(max - min)

        #soooo basically we generate the fake data
        G = netD(fake)
        G = G.mean()
        G += F.mse_loss(fake, improved_fakes)
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        print(G_cost)

    # Write logs and save samples
    # if iteration % 100 == 99:
    generate_image(_data)
    plt.show()

