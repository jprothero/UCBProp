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

#   def __init__(self, input_size, h_dims=400):
#         super(BalancedAE, self).__init__()
#         self.encoder = nn.Parameter(torch.rand(input_size, h_dims))

#     def forward(self, x):
#         x = torch.sigmoid(torch.mm(self.encoder, x))
#         x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
#         return x


class MetaLearner(nn.Module):
    def __init__(self, model, image_size, param_bottleneck=200,
                 h_dims=256, ae_dims=5, num_slices=1000, c=1, use_value=False):
        super(MetaLearner, self).__init__()
        self.use_value = False
        self.num_slices = num_slices
        self.c = c
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

            # in_layers = nn.Parameter(torch.rand(dims, param_bottleneck))

            in_layers = ResBlock(dims, param_bottleneck,
                                 out_dims=h_dims, linear=True)

            out_layers = ResBlock(ae_dims, param_bottleneck,
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

        # self.encoder = nn.Parameter(torch.rand(total_dims, ae_dims))

        self.encoder = ResBlock(h_dims*self.d, ae_dims, linear=True)

        self.value = ValueHead(ae_dims, h_dims, self.d, 1)

        # res_blocks = [ResBlock(9216, h_dims, linear=True)]
        # res_blocks.extend([ResBlock(h_dims, h_dims, linear=True)
        #                    for _ in range(num_hidden_layers-1)])

        # self.res_blocks = nn.Sequential(*res_blocks)

        # self.value = ValueHead(self.d*h_dims, h_dims, self.d)

        # self.group_ins = AttrProxy(self, "group_in_")
        # self.group_outs = AttrProxy(self, "group_out_")

        self.group_ins = nn.ModuleList(self.group_ins)

        # self.encoder = nn.Parameter(torch.rand(param_bottleneck*self.d, ae_dims))

        self.group_outs = nn.ModuleList(self.group_outs)

        self.optim = optim.Adam(self.parameters(), lr=.01, weight_decay=1e-6)
        # well no actually we need a different size for each out

    # idea:
    # what if we use MCTSnet instead of alpha zero
    # MCtsnet has more flexibility, in that you just need to input a state, and it will output
    # search probas
    # now in the paper they still only use a policy
    # but in theory you can probably use a state
    # but what would the state be?
    # basically just the set of parameters like here
    # and we would update some type of embedding for it
    # theres no way to train it if this thing doesnt work though....
    # I dont know a good way.
    # I was hoping this would work
    # it may work with a bandit thing, I do have a feeling that a major issue is that
    # alpha zero is designed for trees and what I'm doing is basically a bandit algo with
    # a shifting distribution
    # for now lets shift gears and try to do the ENAS stuff though.
    # I think that is a lot more promising

    def forward(self, data, target):
        # so lets see... I have one number that in theory dictates all of the parameters(?lol)

        # so if I do an alphazero tweaking it can I optimize that?
        # basically I would need to continually make choices for it

        # self.c = 1000
        # self.lr = .01

        # if train:
        #     self.train()
        # else:
        #     self.eval()

        # self.model.eval()
        # okay so what is the flow. we connect the meta learner to a model
        # we get the performance with the original parameters,
        # no, we get the performance of the parameters with the modified parameters
        # we regress the value head against the accuracy
        # we regress the parameter update against the alphazero improved search (t.b.implemeneted)
        # use the improved parameters, and optionally proceed with normal training, aka fine tune

        self.orig_params = []
        # group_in_stack = [self.image(inp.view(inp.shape[0], -1))]

        group_in_stack = []

        for group, group_in in zip(list(self.model.parameters()), self.group_ins):
            self.orig_params.append(group.view(-1).clone())
            x = group_in(group.view(1, -1)).squeeze()

            group_in_stack.append(x)

        self.orig_flattened_params = torch.cat(self.orig_params)

        group_in_stack = torch.cat(group_in_stack).view(1, -1)

        encoded = self.encoder(group_in_stack).squeeze()

        new_flattened_params = []
        for group_out in self.group_outs:
            new_flattened_params.append(group_out(encoded).squeeze())

        new_flattened_params = torch.cat(new_flattened_params)

        self.optim.zero_grad()
        reconstruction_loss = F.mse_loss(
            new_flattened_params, self.orig_flattened_params)

        # add alphazero stuff here

        root_node = {
            "children": None, "parent": None, "N": 2
        }

        import sys
        # soooo for the value we either need a QRNN
        # or we need to look at the encoded and look at the value for it

        self.linspace = np.linspace(0, 1, self.num_slices)

        from tqdm import tqdm
        value_loss = 0
        
        new_encoded = encoded.clone()

        num_sims = 100
        real_idx = 0
        while real_idx < len(encoded):
            for _ in tqdm(range(num_sims)):
                def do_sim(root_node, value_loss, use_value=False):
                    encoded_copy = encoded.clone()
                    curr_node = root_node
                    exit_early = False
                    prior_idx = 0
                    while curr_node["children"] is not None:
                        if prior_idx < (len(encoded_copy)-1):
                            max_uct = -1e7
                            max_uct_idx = None
                            for i, child in enumerate(curr_node["children"]):
                                child["U"] = (
                                    self.c*child["P"]*np.log(curr_node["N"]))/(1 + child["N"])
                                if (child["Q"] + child["U"]) > max_uct:
                                    max_uct_idx = i
                            curr_node = curr_node["children"][max_uct_idx]
                            encoded_copy[prior_idx] = self.linspace[max_uct_idx]
                            prior_idx += 1
                        else:
                            exit_early = True
                            break

                    if not exit_early:
                        prior = encoded_copy[prior_idx].detach().data.numpy()
                        if not use_value:
                            for group_out. group in zip(self.group_outs, list(self.model.parameters())):
                                group.data = group_out(
                                    encoded_copy).resize_as(group)
                            self.model.eval()
                            out = self.model(data)
                            pred = out.data.max(1, keepdim=True)[1]
                            batch_correct = pred.eq(
                                target.data.view_as(pred)).long().cpu().sum().float()
                            true_value = ((batch_correct/len(target)) - .5)*2
                            value_pred = self.value(encoded_copy)
                            value_loss += F.mse_loss(value_pred, true_value)
                            value = true_value
                        else:
                            value = self.value(encoded_copy).squeeze()
                        linspace_prior = np.log(np.abs(self.linspace - prior + 1e-7))

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

                        curr_node["N"] += 1
                        root_node = curr_node
                    else:
                        while curr_node["parent"] is not None:
                            curr_node = curr_node["parent"]

                        root_node = curr_node

                    return root_node, value_loss
                root_node, value_loss = do_sim(root_node, value_loss, use_value=False)
                # for _ in range(1):
                #     root_node, value_loss = do_sim(root_node, value_loss, use_value=True)

            

            visits_idx = np.argmax([child["N"]
                                   for child in root_node["children"]])


            new_encoded[real_idx] = self.linspace[visits_idx]
            root_node = root_node["children"][visits_idx]
            root_node["parent"] = None
            real_idx += 1

        # encoded_idx = 0
        # curr_node = root_node
        # encoded_copy = encoded.clone()
        # while curr_node["children"] is not None:
        #     visits_idx = np.argmax(child["N"]
        #                            for child in curr_node["children"])
        #     encoded_copy[encoded_idx] = self.linspace[visits_idx]
        #     curr_node = curr_node["children"][visits_idx]
        #     encoded_idx += 1

        improved_encoding_loss = F.mse_loss(encoded, new_encoded)
        total_loss = improved_encoding_loss + reconstruction_loss*.6
        #+ value_loss 
            #
        total_loss.backward()
        self.optim.step()

        for group_out. group in zip(self.group_outs, list(self.model.parameters())):
            group.data = group_out(new_encoded).resize_as(group)

        print(total_loss.detach().data.numpy())

    #     child_priors[:] = self.linspace
    #     child_priors = np.log(
    #         np.abs(np.expand_dims(x.detach().data.numpy(), -1) - child_priors)+1e-7)

        # so what do I do. everytime we try a new bucket we expand
        # but basically we should do_dims for encoded

        # for e in encoded:

        # x = torch.tanh(torch.mm(self.encoder, group_in_stack))
        # x = torch.tanh(torch.mm(x, torch.transpose(self.encoder, 0, 1)))

        # so let me see I basically want an encoder which gets the parameters into a very dense format
        # check, then we need to have an outer group, i.e. the group outs which turns the dense middle back into
        # parameters

        # group_in_stack = x.resize_as(group_in_stack)

        # reconstructed_params = []
        # for x, g_in in zip(group_in_stack, self.group_ins):
        #     reconstructed_params.append(torch.tanh(torch.mm(x, torch.transpose(g_in, 0, 1))).squeeze())

        # reconstructed_params = torch.cat(reconstructed_params)

        # self.optim.zero_grad()
        # reconstruction_loss = F.mse_loss(reconstructed_params, self.orig_flattened_params)
        # reconstruction_loss.backward()
        # self.optim.step()

        # print(reconstruction_loss.detach().data.numpy())

        # return x

        # x = self.res_blocks(group_in_stack)

        # group_out_stack = []
        # for group_out in self.group_outs:
        #     out = group_out(x)
        #     group_out_stack.append(out.squeeze())

        # self.group_out_stack = group_out_stack

        # x = self.tanh(torch.cat(group_out_stack).view(1, -1)).squeeze()

        # out, batch_accuracy = self.do_sims(x, train=train, use_value_head=use_value_head)

        # batch_accuracy /= 2
        # batch_accuracy += .5
        # print("Acc diff: {}".format(batch_accuracy - self.orig_batch_acc))

        # if model_finetune:
        #     self.model.train()
        #     out, batch_accuracy = self.test()
        #     self.model_optim = optim.SGD(self.model.parameters(), lr = .01, momentum=.1)
        #     self.model_optim.zero_grad()
        #     model_loss = F.nll_loss(out, target)
        #     model_loss.backward()
        #     self.model_optim.step()
        #     out, batch_accuracy = self.test(scale=False)
        #     print("Post finetune acc: {}".format(batch_accuracy))

        # #so let me see, right now the prior is x, which is the gated update

        # return out, batch_accuracy

    def _setup_uct_tensors(self, x, num_slices, c):
        self.parent_visits = 2
        self.num_slices = num_slices
        self.flat_shp = x.shape[0]
        self.linspace = linspace = np.linspace(0, 1, num_slices)
        self.child_stats = np.zeros(shape=((self.flat_shp,
                                            len(linspace), 4)))

        self._add_prior(x, c)

    # soooo if we were doing everything with indices we could probably have a key index like -1
    # which indicates it hasnt been expanded, but otherwise we index will point to some other
    # node tensor.

    # what benefit will this give us?
    # the whole point is to make sharing of info easier
    # if a single process is only modifying the list of nodes, it will probably be safe
    # but how do we do that, what will the flow be?

    # we have a shared, expanding tensor which will hold the indices,
    # i.

    #... for simplicity let me just make this really simple to see if this even works

    # for simplicity 0 will be the keyword indicating that a child is not expanded
    # def _setup_root_tensor(self, x):
    #     self.parent_visits = 2
    #     self.linspace = np.linspace(0, 1, self.num_slices)
    #     #so how many stats do we need for each node.
    #     #parent will keep track of the indices
    #     #we need N, W, Q, U, P
    #     #so we also need something that will point to the children
    #     #the children need to be identical tensors to this one
    #     #the number of children will equal the number of lin_spaces
    #     #so the root itself is basically holding the children
    #     #then dynamically we need the children to point to another node
    #     self.indices_tensor = np.zeros(1)
    #     self.root = np.zeros(shape=((self.num_slices, 5)))

    #     self.curr = root
    #     self._add_prior_to_curr(x)

    # def _add_prior_to_curr(self):

    #     #sooo lets see the
    #     np.log(np.abs())

    # def _add_prior(self, x, c, alpha=1, eps=1e-7):
    #     # noise = (np.random.dirichlet([alpha] * self.flat_shp*self.num_slices)-.5)*2
    #     # noise = noise.reshape((self.flat_shp, self.num_slices))
    #     child_priors = np.zeros((self.flat_shp, self.num_slices))
    #     child_priors[:] = self.linspace
    #     child_priors = np.log(
    #         np.abs(np.expand_dims(x.detach().data.numpy(), -1) - child_priors)+1e-7)
    #     child_priors /= np.expand_dims(np.sum(child_priors, axis=1), -1)
    #     noise = np.random.uniform(size=(self.flat_shp, self.num_slices))
    #     child_priors = child_priors*(1-eps) + noise*eps

    #     child_priors = np.expand_dims(child_priors, -1)
    #     child_stats = self.child_stats
    #     child_stats = np.concatenate((child_stats, child_priors), axis=-1)
    #     child_stats[:, :, 3] = c * child_stats[:, :, 4]*(np.log(self.parent_visits) *
    #                                                      (1/(1 + child_stats[:, :, 0])))

    #     self.child_stats = child_stats

    def get_group_ins(self, x):
        results = []
        start_idx = 0
        for g_in, orig_param in zip(self.group_ins, self.orig_params):
            group = x[start_idx:orig_param.data.view(
                -1).shape[0]].resize_as(orig_param).unsqueeze(0)
            results.append(g_in(group))

        results = torch.cat(results).view(1, -1)

        return results

    def get_group_outs(self, x):
        results = []
        for g_out in self.group_outs:
            results.append(g_out(x))

        return torch.cat(results)

    def do_sims(self, encoded):
        self._setup_root_tensor()
        if train:
            out, batch_accuracy = self.test()
            self.optim.zero_grad()
            value = self.value(self.get_group_ins(
                self.orig_flattened_params)).squeeze()
            value_loss = F.mse_loss(value, batch_accuracy)
            batch_accuracy = (batch_accuracy/2) + .5
            self.orig_batch_acc = batch_accuracy
            # print("Orig acc: {}".format(batch_accuracy))

        for _ in range(num_sims):
            indices = self.get_uct_indices(visits=False)
            update = (self.linspace[indices] - .5)*2  # *self.lr
            if use_value_head:
                new_params = update  # self.orig_flattened_params.clone() + update
                value = self.value(self.get_group_ins(new_params))
            else:
                self.update_parameters(update)
                self.model.eval()
                _, value = self.test()
                self.update_parameters(reset=True)

            self.backup_step(indices, value)

        for _ in range(num_sims*5):
            indices = self.get_uct_indices(visits=False)
            update = (self.linspace[indices] - .5)*2  # *self.lr
            update = torch.from_numpy(update).float()
            new_params = update  # self.orig_flattened_params.clone() + update
            value = self.value(self.get_group_ins(new_params)
                               ).squeeze().detach().data.numpy()

            self.backup_step(indices, value)

        indices = self.get_uct_indices(visits=True)
        update = (self.linspace[indices] - .5)*2  # *self.lr
        self.update_parameters(update)
        self.model.eval()
        out, batch_accuracy = self.test()
        # so one of the issues is that all of the parameters have a similar prior
        # they will likely all visit the same choices at the same tie
        # if they are synchonized like that it wont really do anything
        # so how can we fix that.
        # it would be good to have a random distance from the starting prior
        # maybe just mix some dirichlet noise will all of it.

        if train:
            self.train()
            x = self.tanh(
                torch.cat(self.group_out_stack).view(1, -1)).squeeze()
            update_target = torch.from_numpy(update).float()
            param_loss = F.mse_loss(x, update_target)
            new_params = self.orig_flattened_params.clone() + update_target
            value = self.value(self.get_group_ins(new_params)).squeeze()
            value_loss += F.mse_loss(value, batch_accuracy)
            total_loss = param_loss + value_loss
            total_loss.backward()
            self.optim.step()

        # soo we could optionally also have a normal training step for the model,
        # i.e. we take the loss with the improved parameters and move one step with them
        # that might help fine tune

        return out, batch_accuracy

    def update_parameters(self, update=None, reset=False):
        start_idx = 0
        orig_params = self.orig_flattened_params
        for group in list(self.model.parameters()):
            if not reset:
                group.data = torch.from_numpy(
                    update[start_idx:start_idx+group.data.view(-1).shape[0]]).float().resize_as(group)
            else:
                group.data = orig_params[start_idx:start_idx +
                                         group.data.view(-1).shape[0]].resize_as(group)
            start_idx += group.data.view(-1).shape[0]

    def backup_step(self, indices, reward):
        view = self.child_stats[range(len(indices)), indices]
        view[:, 0] += 1  # visits
        view[:, 1] += reward  # batch accuracy
        view[:, 2] = view[:, 1]/view[:, 0]  # Q = W/N
        # can do the above with all of the updated indices rather than individually
        view[:, 3] = self.c * view[:, 4] * \
            (np.log(self.parent_visits)*(1/(view[:, 0])))
        self.child_stats[range(len(indices)), indices] = view

        # so lets see... parent number of visits grows at log, so it's pretty slow
        # Q will always be between 0 and 1, and usually will probably be around .1 to start
        # so the net will focus on good priors, i.e. what the network originally predicted
        # as the net visits most of the options the righ....

        # let me try tweaking the c

        self.parent_visits += 1

    def get_uct_indices(self, visits=False):
        if visits:
            # N
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

# class BalancedAE(nn.module):
#     def __init__(self, input_size, h_dims=400):
#         super(BalancedAE, self).__init__()
#         self.encoder = nn.Parameter(torch.rand(input_size, h_dims))

#     def forward(self, x):
#         x = torch.sigmoid(torch.mm(self.encoder, x))
#         x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
#         return x
