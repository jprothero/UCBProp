from torch import nn
import torch
from ipdb import set_trace
import numpy as np
from torch import optim
import torch.nn.functional as F

# class AttrProxy(object):
#     """Translates index lookups into attribute lookups."""

#     def __init__(self, module, prefix):
#         self.module = module
#         self.prefix = prefix

#     def __getitem__(self, i):
#         return getattr(self.module, self.prefix + str(i))


class MetaLearner(nn.Module):
    def __init__(self, model, image_size, param_bottleneck=100,
                 h_dims=128, num_hidden_layers=6):
        super(MetaLearner, self).__init__()
        self.model = model
        self.tanh = nn.Tanh()
        self.h_dims = h_dims

        self.group_ins = []
        self.group_outs = []

        total_dims = 0
        self.d = 0
        for _, group in enumerate(list(model.parameters())):
            dims = group.view(-1).shape[0]
            total_dims += dims

            in_layers = ResBlock(dims, param_bottleneck,
                                 out_dims=h_dims, linear=True)
            out_layers = ResBlock(h_dims, param_bottleneck,
                                  out_dims=dims, linear=True)

            self.group_ins.append(in_layers)
            self.group_outs.append(out_layers)

            self.d += 1
            # self.add_module("group_in_"+str(i), in_layers)
            # self.add_module("group_out_"+str(i), out_layers)
            # d += 1

        self.CH = CH = image_size[0]
        self.R = R = image_size[1]
        self.C = C = image_size[2]

        self.image = ResBlock(R*C*CH, h_dims, linear=True)

        res_blocks = [ResBlock(9216, h_dims, linear=True)]
        res_blocks.extend([ResBlock(h_dims, h_dims, linear=True)
                           for _ in range(num_hidden_layers-1)])

        self.res_blocks = nn.Sequential(*res_blocks)

        self.value = ValueHead(self.d*h_dims, h_dims, self.d)

        # self.group_ins = AttrProxy(self, "group_in_")
        # self.group_outs = AttrProxy(self, "group_out_")

        self.group_ins = nn.ModuleList(self.group_ins)
        self.group_outs = nn.ModuleList(self.group_outs)

        self.optim = optim.SGD(self.parameters(), lr=.1, momentum=.9, weight_decay=1e-6)
        # well no actually we need a different size for each out

    def forward(self, inp, target, train=False, num_sims=200, num_slices=5, c=2, 
            model_finetune=True, use_value_head=False):
        self.inp = inp
        self.target = target
        self.c = 1000
        self.lr = .01

        if train:
            self.train()
        else:
            self.eval()

        self.model.eval()
        # okay so what is the flow. we connect the meta learner to a model
        # we get the performance with the original parameters,
        # no, we get the performance of the parameters with the modified parameters
        # we regress the value head against the accuracy
        # we regress the parameter update against the alphazero improved search (t.b.implemeneted)
        # use the improved parameters, and optionally proceed with normal training, aka fine tune

        self.orig_params = []
        group_in_stack = [self.image(inp.view(inp.shape[0], -1))]

        for group, group_in in zip(list(self.model.parameters()), self.group_ins):
            self.orig_params.append(group.view(-1).clone())
            group_in_stack.append(group_in(group.view(1, -1)))

        self.orig_flattened_params = torch.cat(self.orig_params)

        group_in_stack = torch.cat(group_in_stack).view(1, -1)

        x = self.res_blocks(group_in_stack)

        group_out_stack = []
        for group_out in self.group_outs:
            out = group_out(x)
            group_out_stack.append(out.squeeze())

        self.group_out_stack = group_out_stack

        x = self.tanh(torch.cat(group_out_stack).view(1, -1)).squeeze()

        out, batch_accuracy = self.do_sims(x, train=train, use_value_head=use_value_head)

        batch_accuracy /= 2
        batch_accuracy += .5
        print("Acc diff: {}".format(batch_accuracy - self.orig_batch_acc))

        if model_finetune:
            self.model.train()
            out, batch_accuracy = self.test()
            self.model_optim = optim.SGD(self.model.parameters(), lr = .01, momentum=.1)
            self.model_optim.zero_grad()
            model_loss = F.nll_loss(out, target)
            model_loss.backward()
            self.model_optim.step()
            out, batch_accuracy = self.test(scale=False)
            print("Post finetune acc: {}".format(batch_accuracy))        

        #so let me see, right now the prior is x, which is the gated update

        return out, batch_accuracy

    def _setup_uct_tensors(self, x, num_slices, c):
        self.parent_visits = 2
        self.num_slices = num_slices
        self.flat_shp = x.shape[0]
        self.linspace = linspace = np.linspace(0, 1, num_slices)
        self.child_stats = np.zeros(shape=((self.flat_shp, 
            len(linspace), 4)))

        self._add_prior(x, c)

    def _add_prior(self, x, c, alpha=1, eps=1e-7):
        # noise = (np.random.dirichlet([alpha] * self.flat_shp*self.num_slices)-.5)*2
        # noise = noise.reshape((self.flat_shp, self.num_slices))
        child_priors = np.zeros((self.flat_shp, self.num_slices))
        child_priors[:] = self.linspace
        child_priors = np.log(
            np.abs(np.expand_dims(x.detach().data.numpy(), -1) - child_priors)+1e-7)
        child_priors /= np.expand_dims(np.sum(child_priors, axis=1), -1)
        noise = np.random.uniform(size=(self.flat_shp, self.num_slices))
        child_priors = child_priors*(1-eps) + noise*eps

        child_priors = np.expand_dims(child_priors, -1)
        child_stats = self.child_stats
        child_stats = np.concatenate((child_stats, child_priors), axis=-1)
        child_stats[:, :, 3] = c * child_stats[:, :, 4]*(np.log(self.parent_visits) *
                                                         (1/(1 + child_stats[:, :, 0])))

        self.child_stats = child_stats

    def get_group_ins(self, x):
        results = []
        start_idx = 0
        for g_in, orig_param in zip(self.group_ins, self.orig_params):
            group = x[start_idx:orig_param.data.view(-1).shape[0]].resize_as(orig_param).unsqueeze(0)
            results.append(g_in(group))

        results = torch.cat(results).view(1, -1)

        return results

    def get_group_outs(self, x):
        results = []
        for g_out in self.group_outs:
            results.append(g_out(x))

        return torch.cat(results)

    def do_sims(self, x, num_sims=10, num_slices=100, c=1, train=False, use_value_head=False):
        self._setup_uct_tensors(x, num_slices, c)
        if train:
            out, batch_accuracy = self.test()
            self.optim.zero_grad()
            value = self.value(self.get_group_ins(self.orig_flattened_params)).squeeze()
            value_loss = F.mse_loss(value, batch_accuracy)    
            batch_accuracy = (batch_accuracy/2) + .5    
            self.orig_batch_acc = batch_accuracy
            # print("Orig acc: {}".format(batch_accuracy))

        for _ in range(num_sims):
            indices = self.get_uct_indices(visits=False)
            update = (self.linspace[indices] - .5)*2#*self.lr
            if use_value_head:
                new_params = update # self.orig_flattened_params.clone() + update            
                value = self.value(self.get_group_ins(new_params))
            else:
                self.update_parameters(update)
                self.model.eval()
                _, value = self.test()
                self.update_parameters(reset=True)                

            self.backup_step(indices, value)

        for _ in range(num_sims*5):
            indices = self.get_uct_indices(visits=False)
            update = (self.linspace[indices] - .5)*2#*self.lr
            update = torch.from_numpy(update).float()
            new_params = update #self.orig_flattened_params.clone() + update            
            value = self.value(self.get_group_ins(new_params)).squeeze().detach().data.numpy()

            self.backup_step(indices, value)

        indices = self.get_uct_indices(visits=True)
        update = (self.linspace[indices] - .5)*2#*self.lr
        self.update_parameters(update)
        self.model.eval()
        out, batch_accuracy = self.test()
        #so one of the issues is that all of the parameters have a similar prior
        #they will likely all visit the same choices at the same tie
        #if they are synchonized like that it wont really do anything
        #so how can we fix that.
        #it would be good to have a random distance from the starting prior
        #maybe just mix some dirichlet noise will all of it.

        if train:
            self.train()
            x = self.tanh(torch.cat(self.group_out_stack).view(1, -1)).squeeze()
            update_target = torch.from_numpy(update).float()
            param_loss = F.mse_loss(x, update_target)
            new_params = self.orig_flattened_params.clone() + update_target            
            value = self.value(self.get_group_ins(new_params)).squeeze()
            value_loss += F.mse_loss(value, batch_accuracy)
            total_loss = param_loss + value_loss
            total_loss.backward()
            self.optim.step()
    
        #soo we could optionally also have a normal training step for the model,
        #i.e. we take the loss with the improved parameters and move one step with them
        #that might help fine tune

        return out, batch_accuracy

    def update_parameters(self, update=None, reset=False):
        start_idx = 0
        orig_params = self.orig_flattened_params
        for group in list(self.model.parameters()):
            if not reset:
                group.data = torch.from_numpy(update[start_idx:start_idx+group.data.view(-1).shape[0]]).float().resize_as(group)
            else:
                group.data = orig_params[start_idx:start_idx+group.data.view(-1).shape[0]].resize_as(group)
            start_idx += group.data.view(-1).shape[0]

    def test(self, scale=True):
        out = self.model(self.inp)
        pred = out.data.max(1, keepdim=True)[1]
        batch_correct = pred.eq(
            self.target.data.view_as(pred)).long().cpu().sum().float()
        batch_accuracy = batch_correct/len(self.target)
        if scale:
            batch_accuracy = (batch_accuracy - .5)*2

        return out, batch_accuracy

    def backup_step(self, indices, reward):
        view = self.child_stats[range(len(indices)), indices]
        view[:, 0] += 1 #visits
        view[:, 1] += reward #batch accuracy
        view[:, 2] = view[:, 1]/view[:, 0] #Q = W/N
            #can do the above with all of the updated indices rather than individually
        view[:, 3] = self.c * view[:, 4]*(np.log(self.parent_visits)*(1/(view[:, 0])))
        self.child_stats[range(len(indices)), indices] = view

        #so lets see... parent number of visits grows at log, so it's pretty slow
        #Q will always be between 0 and 1, and usually will probably be around .1 to start
        #so the net will focus on good priors, i.e. what the network originally predicted
        #as the net visits most of the options the righ....

        #let me try tweaking the c

        self.parent_visits += 1

    def get_uct_indices(self, visits=False):
        if visits:
            #N 
            uct_view = self.child_stats[:, :, 0]
            uct_view = np.exp(uct_view)
            uct_view /= np.expand_dims(np.sum(uct_view, axis=1), -1)
        else:
            #Q + U
            uct_view = self.child_stats[:, :, 2] + self.child_stats[:, :, 3]
        
        indices = np.argmax(uct_view, axis=1)
        return indices

class ResBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=None, linear=True):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        if linear:
            self.layer1 = nn.Linear(in_dims, h_dims)
            self.n1 = nn.LayerNorm(h_dims)
        # else:
        #     self.layer1 = nn.Conv2d(in_dims, h_dims, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm2d(h_dims)

        if out_dims is None:
            out_dims = h_dims

        if linear:
            self.layer2 = nn.Linear(h_dims, out_dims)
            self.n2 = nn.LayerNorm(out_dims)
        # else:
        #     self.layer2 = nn.Conv2d(h_dims, out_dims, kernel_size=1, stride=1)
        #     self.bn2 = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        residual = x.squeeze()

        out = self.layer1(x)

        out = self.n1(out)
        out = self.relu(out)

        out = self.layer2(out)
        out = self.n2(out)

        if out.shape == residual.shape:
            out += residual
        out = self.relu(out)

        return out


class ValueHead(nn.Module):
    def __init__(self, in_dims, h_dims, d, out_dims=1):
        super(ValueHead, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.lin1 = nn.Linear(in_dims, h_dims)
        self.n1 = nn.LayerNorm(h_dims)
        self.lin2 = nn.Linear(h_dims, 32)

        self.scalar = nn.Linear(32, 1)

    def forward(self, inp):
        x = self.lin1(inp)
        x = self.n1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.relu(x)

        x = self.scalar(x)
        value = self.tanh(x)

        return value
