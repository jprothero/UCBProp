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
        self.child_stats = self.create_stats_tensor()


        # self.curr_params = self.get_parameters(view)
    
    def create_stats_tensor(self):
        self.parent_visits = 2        
        # child_priors = np.zeros((len(self.params), len(self.linspace)))
        # child_priors[:] = self.linspace
        # child_priors = np.log(np.abs(self.params - child_priors))
        # child_priors /= np.expand_dims(np.sum(child_priors, axis=1), -1)
        # child_priors = np.expand_dims(child_priors, -1)

        child_stats = np.zeros(shape=(len(self.params), len(self.linspace), 3))
        # child_stats = np.concatenate((child_stats, child_priors), axis=-1)
        # child_stats[:, :, 3] = self.c * child_stats[:, :, 5] * \
        #      (np.log(self.parent_visits)*(1/(1 + child_stats[:, :, 0])))

        # child_stats[:, :, 3] /= np.expand_dims(np.sum(child_stats[:, :, 3], axis=1), -1)

        child_stats[:, :, 0] += 1
        child_stats[:, :, 1] += .01
        child_stats[:, :, 2] += child_stats[:, :, 1] / child_stats[:, :, 0]

        return child_stats

    def get_uct_indices(self):
        #update UCT scores
        # set_trace()
        #so lets see, we want the more that Q is close to 1 the less random it is
        #is in theory if all Q's 0 1 we have 0 exploration
        #right now we have the raw Q values, we get those and choose whether to
        #take that argmax or not based on if Q - (0-1) > 0
        #so that will happen only like ~30% of the time at first
        #it should probaby be more.
        #and also the difference between have .3 accuracy and .1 should be big, so we want
        #to square or cube q
        #also we probably want a moving average of what the average score is
        #so the uniform's max is the current moving average
        q_view = self.child_stats[:, :, 2]**10
        #argmax Q's
        q_indices = np.argmax(q_view, axis=1)
        q_indices_values = q_view[range(q_view.shape[0]), q_indices]
        #so I want the current average of the q_values
        # average_q = np.mean(q_indices_values) #lower it
        # max_q = np.max(q_indices_values) #lower it

        #so if the Q value is below average it will always be explored
        #otherwise it wont, one issue is that 

        random = np.random.uniform(low = 0, high=1, size=q_indices.shape)

        #so we could add a c to control how much c is 
        u_view = (self.child_stats[:, :, 2]) / (2 + self.child_stats[:, :, 0])
        # u_view /= np.expand_dims(np.sum(u_view, axis=1), 1)
        u_indices = np.argmax(u_view, axis=1)
        u_indices_values = u_view[range(u_view.shape[0]), u_indices]
        # average_u = np.mean(u_indices_values)    
        # max_u = np.max(u_indices_values)

        # average_u = np.mean(u_indices_values)        
        random2 = np.random.uniform(low=0, high=1,size=u_view.shape)
        u_view -= random2
        u_indices = np.argmax(u_view, axis=1)

        indices = np.where((q_indices_values - random) > 0, q_indices, u_indices)

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
        view[:, 3] = self.c * view[:, 5] * \
            (np.log(self.parent_visits)*(1/(view[:, 0])))
        #view[:, 3] = view[:, 2]
            #can do the above with all of the updated indices rather than individually

        #view[:, 3] = #so we probably want the exploration to be relative to the average value
        #so it will be 1 + Q

        #so again we want the right hand side to be between 0 and 1 ideally
        #Q is between 0 and 1
        #1 - Q will be between 0 and 1
        #we want to pick the choice which is 
        #so maybe we alternate between picking by U score and by Q score
        #so the Q score is the probability we pick the Q score
        #Q score is how it is now
        #U score should be some chance thing too
        #so the chance of exploring an option is based on it's Q score, so .Q chance of 
        #choosing that choice, we maybe need to normalize that

        # np.log(self.parent_visits)*(1/(view[:, 0]))

        # *view[:, 5] * \

        self.child_stats[range(len(indices)), indices] = view

        #so lets see the reward is going to be between 0 and 1
        #we need the right hand to be between 0 and 1
        #so do we want it divided by num visits?
        #basically we want it so that unexplored ones (low visits)
        #have a higher value
        #

        # self.parent_visits += 1
        # self.parent_visits = self.parent_visits % 10
        # self.c += 5
        # self.c = self.c % 50

        #what do we want on the right side
        #it needs to be between 0 and 1, and basically the more explorations the less value
        #i.e. / by N
        #we want exploration to be an empirically chosen constant, or a cylical thing,
        #so maybe take away the log (parents all together)
        #we need exploration factor to oscilate between 0 and 1
        #so the closer to 1 the average reward is the less it will explore
        #

        #so in theory the more times we visits a node the lower the exploration value
        #the issue here is that the exploitation side is always winning out
        #so overtime parent visits is slowly increasing, i.e. slowing raising the value
        #of exploring

        #the bottom decreases the exploration value based on how many visits there are
        #that is the main issue, is that we are visits all of the options a lot and their
        #percentages get decreased a lot
        #can I slow the growth of the bottom number, maybe log of it?

        #so this will oscilate between 0 and 10

        self.curr_params = self.get_parameters(indices, update_params)

    def update_model(self):
        for i, param in enumerate(self.model.parameters()):
            param.data = self.curr_params[i]
            # assert (param.data == self.curr_params[i]).all()

    def step(self, reward, update_params=False):
        self.update_nodes(reward, update_params)
        # if self.count % self.reset_every == 0 and self.count != 0:
            # self.params = np.expand_dims(self.params, -1)+1e-7
            # self.child_stats = self.create_stats_tensor()
        self.update_model()
        

        #so what is the issue. this system gets a lot of visits since it 
        #is a bander
        

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


        



        