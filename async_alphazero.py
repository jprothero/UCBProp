import numpy as np
import torch
from torch.nn import Parameter
from ipdb import set_trace

class AsyncAlphazero:
    def __init__(self, model, num_slices=10, c=1, cycles_per_batch=10, num_sims=50, lr=.01,
        num_steps=10):
        self.model = model
        self.orig_orig_params = list(model.parameters())        
        self.orig_params = list(model.parameters())
        self.flattened_param_sizes = []

        self.num_sims = num_sims

        self.num_steps = num_steps

        self.lr = lr

        self.c = c       

        self.count = 0

        self.cycles_per_batch = cycles_per_batch

        self.num_slices = num_slices
        self.linspace = np.linspace(0, 1, self.num_slices)

        temp = []
        for param in self.orig_params:
            temp.append(param.view(-1))
            self.flattened_param_sizes.append(param.view(-1).size())

        params_shape = torch.cat(temp, dim=0).detach().data.numpy().shape
        self.flattened_params = np.random.uniform(size=params_shape) -.5
        self.orig_flattened_params = np.copy(self.flattened_params)
        
        #need to shift these back by -.5 at the end to mean 0 them
        # self.params = np.ones(shape=(flattened_params.shape[0], 1))
        # self.params = np.zeros(shape=(flattened_params.shape[0], 1))#+\
            # (1/flattened_params.shape[0])

        self.reset_az()

        #initialize to 2 so that UCT works on the first step
        
        self.trajectory_indices = []

        # self.curr_params = self.get_parameters(view)
    
    def reset_az(self):
        self.parent_visits = 2
        self.step_nodes = [self.create_stats_tensor() for _ in range(self.num_steps)]

    def create_stats_tensor(self):
        child_priors = np.zeros((len(self.flattened_params), len(self.linspace)))
        child_priors[:] = self.linspace
        child_priors = np.log(np.abs(np.expand_dims(self.flattened_params, -1)  - child_priors))
        child_priors /= np.expand_dims(np.sum(child_priors, axis=1), -1)
        child_priors = np.expand_dims(child_priors, -1)

        child_stats = np.zeros(shape=((len(self.flattened_params), len(self.linspace), 4)))
        child_stats = np.concatenate((child_stats, child_priors), axis=-1)
        child_stats[:, :, 3] = self.c * child_stats[:, :, 4]*(np.log(self.parent_visits)*(1/(1 + child_stats[:, :, 0])))

        return child_stats

    def get_uct_indices(self, step_node, visits = False):
        #update UCT scores
        # set_trace()
        #view just the UCT scores
        uct_view = step_node[:, :, 2] + step_node[:, :, 3]
        #argmax the max for each param
        if visits:
            uct_view = step_node[:, :, 0]
        indices = np.argmax(uct_view, axis=1)
        return indices

    def update_params_step(self, step_node, visits=False):
        indices = self.get_uct_indices(step_node, visits)
        self.update_model(self.get_parameters(indices))
        if not visits: 
            self.trajectory_indices.append(indices)

    def backup_step(self, reward):
        for i, indices in enumerate(self.trajectory_indices):
            #so what do I want to do, I want the indices for each of the steps
            #to get updated. so I want to save the indices for each step
            view = self.step_nodes[i][range(len(indices)), indices]
            view[:, 0] += 1 #visits
            view[:, 1] += reward #batch accuracy
            view[:, 2] = view[:, 1]/view[:, 0] #Q = W/N
                #can do the above with all of the updated indices rather than individually
            view[:, 3] = self.c * view[:, 4]*(np.log(self.parent_visits)*(1/(view[:, 0])))
            self.step_nodes[i][range(len(indices)), indices] = view

        #uhhh is the parent visits thing right?, I think I need to use the visits 
        #from the prev tracjectory
        self.parent_visits += 1
        self.trajectory_indices = [] 
        self.flattened_params = np.array(self.orig_flattened_params)

        #well let me think about it
        #the parent_visits for all steps will be the same
        #we are actually comparing to the parent vistis so yeah its fine

    def update_model(self, param_groups):
        for i, param in enumerate(self.model.parameters()):
            param.data = param_groups[i]
            # assert (param.data == self.curr_params[i]).all()

    # def step(self, reward, update_params=False):
    #     # self.update_nodes(reward, update_params)
    #     # if self.count % self.reset_every == 0 and self.count != 0:
    #         # self.params = np.expand_dims(self.params, -1)+1e-7
    #         # self.child_stats = self.create_stats_tensor()
    #     self.update_model()
        

        #so what is the issue. this system gets a lot of visits since it 
        #is 

    def get_parameters(self, indices):
        #.-5 makes it mean 0 between -.5 and .5
        # set_trace()        
        #so lets think about the idea to convert to the 
        #number of visits
        self.flattened_params += ((self.linspace[indices] - .5)*self.lr)

        parameters = self.flattened_params
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


        



        