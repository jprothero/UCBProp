import numpy as np
import torch
from torch.nn import Parameter
from ipdb import set_trace

class AsyncAlphazero:
    def __init__(self, model, num_slices=10, c=1, cycles_per_batch=10, num_sims=50, reset_every=20):
        self.model = model
        self.orig_params = list(model.parameters())
        self.flattened_param_sizes = []

        self.num_sims = num_sims

        self.c = c       

        self.count = 0
        self.reset_every = reset_every

        self.cycles_per_batch = cycles_per_batch

        self.num_slices = num_slices
        self.linspace = np.linspace(0, 1, self.num_slices)

        temp = []
        for param in self.orig_params:
            temp.append(param.view(-1))
            self.flattened_param_sizes.append(param.view(-1).size())

        flattened_params = torch.cat(temp, dim=0).detach().data.numpy()
        
        #need to shift these back by -.5 at the end to mean 0 them
        self.params = np.random.uniform(size=(flattened_params.shape[0], 1))
        # self.params = np.ones(shape=(flattened_params.shape[0], 1))
        # self.params = np.zeros(shape=(flattened_params.shape[0], 1))#+\
            # (1/flattened_params.shape[0])

        #initialize to 2 so that UCT works on the first step
        self.parent_visits = 2

        self.child_stats = self.create_stats_tensor()


        # self.curr_params = self.get_parameters(view)
    
    def create_stats_tensor(self):
        child_priors = np.zeros((len(self.params), len(self.linspace)))
        child_priors = self.linspace
        child_priors = np.log(np.abs(self.params - child_priors))
        child_priors /= np.expand_dims(np.sum(child_priors, axis=1), -1)
        child_priors = np.expand_dims(child_priors, -1)

        child_stats = np.zeros(shape=(child_priors.shape[0], child_priors.shape[1], 5))
        child_stats = np.concatenate((child_stats, child_priors), axis=-1)
        child_stats[:, :, 3] = self.c * child_stats[:, :, 5] * \
             (np.log(self.parent_visits)*(1/(1 + child_stats[:, :, 0])))

        return child_stats

    def get_uct_indices(self):
        #update UCT scores
        # set_trace()
        self.child_stats[:, :, 4] = self.child_stats[:, :, 2] + self.child_stats[:, :, 3]
        #view just the UCT scores
        uct_view = self.child_stats[:, :, 4]
        #argmax the max for each param
        indices = np.argmax(uct_view, axis=1)
        return indices

    def update_nodes(self, reward, update_params=False):
        indices = self.get_uct_indices()
        view = self.child_stats[range(len(indices)), indices]
        
        #0=N
        #1=W
        #2=Q
        #3=
        view[:, 0] += 1 #visits
        view[:, 1] += reward #batch accuracy
        view[:, 2] = view[:, 1]/view[:, 0] #Q = W/N
            #can do the above with all of the updated indices rather than individually
        view[:, 3] = self.c * view[:, 5] * \
            (np.log(self.parent_visits)*(1/(view[:, 0])))

        # *view[:, 5] * \

        self.child_stats[range(len(indices)), indices] = view

        self.parent_visits += 1

        self.curr_params = self.get_parameters(indices, update_params)

    def update_model(self):
        for i, param in enumerate(self.model.parameters()):
            param.data = self.curr_params[i]
            # assert (param.data == self.curr_params[i]).all()

    def step(self, reward, update_params=False):
        self.update_nodes(reward, update_params)
        if self.count % self.reset_every == 0 and self.count != 0:
            self.child_stats = self.create_stats_tensor()
        self.update_model()


    def get_parameters(self, indices, update_params=False):
        #.-5 makes it mean 0 between -.5 and .5
        # set_trace()        
        #so lets think about the idea to convert to the 
        #number of visits

        if update_params:
            #UCT scores
            #so how was I doing it, I was
            # self.child_stats[:, :, 0] /= np.expand_dims(np.sum(self.child_stats[:, :, 0], axis=1), -1)
            # visits = self.child_stats[:, :, 0] 
            visits = np.exp(self.child_stats[:, :, 0])
            visits /= np.expand_dims(np.sum(visits, axis=1), -1)
            parameters = (self.linspace * visits)
            parameters = np.sum(parameters, axis=1)
            self.params = np.random.uniform(size=(parameters.shape[0], 1))

            parameters -= .5
            print(parameters)
        else:
            self.params = self.linspace[indices]
            parameters = self.params - .5
            # self.child_stats[:, :, 5] = np.random.uniform(size=(parameters.shape[0], 
            #     len(self.linspace)))
        starting_idx = 0
        param_groups = []
        for i, flattened_param_size in enumerate(self.flattened_param_sizes):
            param_groups.append(torch.from_numpy(parameters[starting_idx:starting_idx+\
                flattened_param_size[0]]))
            param_groups[-1] = param_groups[-1].float().view_as(self.orig_params[i])
            starting_idx = flattened_param_size[0]

        #okay... so that's a huge issue that they are all the same
            
        return param_groups


        



        