import numpy as np
import torch
import torch.nn as nn

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def game_initializer(module):
    classname = module.__class__.__name__
    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()


#Creating one architecture for both the actor an the critic
# Actor output = num_actions, critic output = 1

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # Architecture from the Original DQN paper
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.output_size = 64 * 7 * 7

        self.lin = nn.Sequential(nn.Linear(self.output_size, 512),
                                nn.ReLU())

        self.a = nn.Linear(512, num_actions)
        self.c = nn.Linear(512, 1)

        self.num_actions = num_actions

        # parameter initialization
        self.apply(game_initializer)
        self.a.weight.data = ortho_weights(self.a.weight.size(), scale=.01)
        self.c.weight.data = ortho_weights(self.c.weight.size())

    def forward(self, x):
        #obtaining the right dimension
        bs = x.size()[0]

        conv_out = self.conv(bs).view(bs, self.output_size)

        fc_out = self.lin(conv_out)

        a = self.a(fc_out)
        c = self.c(fc_out)

        return a, c