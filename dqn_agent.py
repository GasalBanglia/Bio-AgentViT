from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



Transition = namedtuple('Transition',
                        ('state', 'new_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self, capacity, buffer_batch_size):
        self.buffer_batch_size = buffer_batch_size
        self.memory = []
        self.capacity = capacity

    def push(self, *args):
        if len(self.memory) >= self.capacity:
            index_to_remove = random.randint(0, len(self.memory) - 1)
            self.memory.pop(index_to_remove)
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.buffer_batch_size)

    def __len__(self):
        return len(self.memory)


'''
class QNetwork(nn.Module):
    def __init__(self, n_patches):
        super(QNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(n_patches, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_patches)
        )
'''

# QNetwork class.
# MLP-based QNetwork from original paper.
# Can take in an input of any size; originally designed to receive attention scores from transformer.
# Has 'n_patches' outputs: the probabilities of a patch being selected 
class QNetwork(nn.Module):
    def __init__(self, input_size, n_patches):  # Add input_size
        super(QNetwork, self).__init__()
        self.fc_layers = nn.Sequential(
                 nn.Linear(input_size, 1024),  # Change input size here
                 nn.ReLU(),
                 nn.Linear(1024, 256),
                 nn.ReLU(),
                 nn.Linear(256, n_patches)
             )

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.fc_layers(input)
        return x

class ConvNet(nn.Module):
    def __init__(self, num_patches):
        super(ConvNet, self).__init__()
        # layers - in_channels are number of channels in the input image
        # out_channels are number of channels in the output image, i.e. the number of kernels we operate on the input image
        # kernel_size is the size of the kernel (e.g. 5 means 5 by 5)
        # the input image size is 32 by 32, so that using the formula (w - f + 2p)/s + 1 as the size of an output channel multiple times we get
        # 5*5 as the dimension of each channel right before the fc1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.out = nn.Linear(84, num_patches)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # the -1 in the following in the number of samples in our batch
        x = x.view(-1, 16*5*5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # x = self.fc3(x) # notice that the last layer doesn't need a softmax activation since that's already included in nn.CrossEntropyLoss()
        x = self.out(x)
        return x        

class QNetworkCNN(nn.Module):
    def __init__(self, input_size, n_patches, pretrained=False, freeze_layers=True):  # Add input_size
        super(QNetworkCNN, self).__init__()
        self.network = ConvNet(n_patches)
        print("ConvNet created! Value of pretrained:", pretrained)
        if pretrained:
            self.network.load_state_dict(torch.load("./models/CNN_CIFAR10_parameters.pth"), strict=False)
            print("model weights loaded!")
        if freeze_layers:  # Added logic for freeze
            # Freeze all layers except the last one (self.out)
            for name, param in self.network.named_parameters():
                if 'out' not in name: # The name parameter allows us to filter layers using their name.
                    param.requires_grad = False
            

        # if pretrained:
        #     self.network.load_state_dict("./models/CNN_CIFAR10_parameters.pth")
        # self.features = nn.Sequential(*list(self.network.children())[:-2])
        # print(self.features)
        # self.patch_selection_head = nn.Linear(84, n_patches)

    def forward(self, input):
        # x = input.view(input.size(0), -1)
        # x = self.fc_layers(input)
        # x = self.features(input)
        # x = x.view(x.size(0), -1)
        # x = self.patch_selection_head(x)
        return self.network(input)
    
# DQNAgent class
# Main class for training and patch selection.
# uses one of the types of Q-Networks defined above.
# Includes methods for updates/optimization
class DQNAgent():
    def __init__(self, buffer_batch_size, att_dim, n_patches, buffer_size, gamma, tau, update_every, lr, env, device, input_size, pretrained=False):
        self.buffer_batch_size = buffer_batch_size
        self.gamma = gamma

        # soft update parameter
        self.tau = tau
        # soft_update frequency
        self.step = 0
        self.update_every = update_every

        # learning rate
        self.lr = lr

        # device (CPU o GPU)
        self.device = device

        # environment
        self.env = env

        # patch width
        self.patch_size = np.sqrt(n_patches)

        # agent net
        #self.q_network = QNetwork(n_patches).to(self.device)
        # target net
        #self.target_network = QNetwork(n_patches).to(self.device)

        # input_size = 16 * 16 * 3
        self.q_network = QNetworkCNN(input_size, n_patches, pretrained=pretrained).to(self.device)
        self.target_network = QNetworkCNN(input_size, n_patches, pretrained=pretrained).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, amsgrad=True)
        # replay memory
        self.memory = ReplayBuffer(buffer_size, buffer_batch_size)


    def select_action(self, data, eps = 0.):

        sample = random.random()

        if sample > eps:
            with torch.no_grad():
                selected_batch = self.q_network(data)
                selected = torch.mean(selected_batch, dim=0)
                return selected
        else:
            selected = self.env.action_space.sample()
            return selected



    def optimize_model(self):
        if len(self.memory) < self.buffer_batch_size:
            return
        transitions = self.memory.sample()

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        new_state_batch = torch.cat(batch.new_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.q_network(state_batch)

        with torch.no_grad():
            next_state_values = self.target_network(new_state_batch)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(1)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        self.step += 1
        if self.step == self.update_every:
            self.soft_update()
            self.step = 0


    def soft_update(self):
        target_network_state_dict = self.target_network.state_dict()
        q_network_state_dict = self.q_network.state_dict()
        for key in q_network_state_dict:
            target_network_state_dict[key] = q_network_state_dict[key]*self.tau + target_network_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_network_state_dict)