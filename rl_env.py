# import torchvision
import time
import torchvision.transforms as transforms
import torch
from torch.nn import functional


import gym

# import torch.utils.data as data
# import torch.nn as nn
# from torch.nn import functional
# import torch.optim as optim

# Action space represented with a vector with one element for every feature
class MultiContinue():

    def __init__(self,  n_patch, device):
        self.n_patch = n_patch
        self.device = device

    def sample(self):
        action = np.random.rand(self.n_patch)
        return torch.tensor(action, device=self.device, dtype=torch.float)


def train_iter_agent(model, optimizer, data, target):
    start_time = time.time()

    model.train()

    optimizer.zero_grad()
    out = functional.log_softmax(model(data), dim=1)
    loss = functional.nll_loss(out, target)
    loss.backward()
    optimizer.step()

    iteration_time = time.time() - start_time

    return loss.item(), iteration_time

def evaluate_agent(model, data_load, device):

    patches = model.get_patches()
    model.set_patches(torch.tensor([1 for i in range(len(patches))], dtype=torch.float))
    model.eval()


    elements = 0
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:

            elements += len(data)
            data = data.to(device)
            target = target.to(device)

            output = functional.log_softmax(model(data), dim=1)
            loss = functional.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()

    loss_val = tloss / elements
    acc_val = (100.0 * csamp / elements).cpu()

    print('\nAverage test loss: ' + '{:.4f}'.format(loss_val) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(elements) + ' (' +
          '{:4.2f}'.format(acc_val) + '%)\n')

    model.set_patches(patches)

    return loss_val, acc_val


def train_validation_agent(model, optimizer, train_data, train_target, validation_loader, device):

  start_time = time.time()

  tr_loss = train_iter_agent(model, optimizer, train_data, train_target)

  val_loss, val_acc = evaluate_agent(model, validation_loader, device)

  iteration_time = time.time() - start_time

  return tr_loss, val_loss, val_acc, iteration_time


class ViTEnv(gym.Env):
    def __init__(self, ViTnet, n_patch, optimizer, loss_weight, time_weight, device, n_patch_selected = 1, seed = None, total_epochs=20):
        super().__init__()

        self.ViTnet = ViTnet
        self.optimizer = optimizer

        self.seed = seed

        self.loss_weight = loss_weight
        self.time_weight = time_weight

        self.action_space = MultiContinue(n_patch, device)

        self.device = device

        self.train_loss_list = []
        self.train_time_list = []

        self.total_epochs = total_epochs
        self.current_epoch = 0  # Initialize current epoch

        self.n_patch_selected = n_patch_selected

    def step_train(self, action, train_data, train_target):

        self.ViTnet.set_patches(action)

        train_iter_agent(self.ViTnet, self.optimizer, train_data, train_target)


    def step_reward(self, action, train_data, train_target):

        self.ViTnet.set_patches(action)

        action = self.ViTnet.get_patches()

        print(f'  Patch list: {action}')
        print(f'  Selected Patches: {action.count(1)}')

        reward = self.get_reward(action, train_data = train_data, train_target = train_target)

        return self.get_state(train_data), reward


    def get_reward(self, action, train_data, train_target):

        train_loss, iteration_time = train_iter_agent(self.ViTnet, self.optimizer, train_data, train_target)

        self.train_time_list.append(iteration_time)
        self.train_loss_list.append(train_loss)

        #num_zeros = action.count(0)
        #ideal_zeros = len(action) - n_patch_selected;
        #patches_reward = (- abs(num_zeros - ideal_zeros) / ideal_zeros)
        #loss_reward = (self.train_loss_list[0]/train_loss)

        num_zeros = action.count(0)
        ideal_zeros = len(action) - self.n_patch_selected
        patches_reward = (- abs(num_zeros - ideal_zeros) / ideal_zeros) # * (1 - self.current_epoch / self.total_epochs)  # Decaying reward
        
        # time_reward = (self.train_time_list[-1] - self.train_time_list[-2]) if len(self.train_time_list) > 1 else self.train_time_list[0]
        # Per the paper: "ratio between the value ℒ(0) of the loss function of the 
        # ViT at the starting iteration and the value ℒ(𝑡) of the same function at iteration 𝑡"
        loss_reward = (self.train_loss_list[0] / train_loss)

        print(f'  loss_reward: {loss_reward}')
        print(f'  patches_reward: {patches_reward}')
        # print(f'  time_reward: {time_reward}')
        reward = loss_reward * self.loss_weight + patches_reward * self.time_weight

        return reward

    # def get_state(self, data):
    #    with torch.no_grad():
    #      return data

    def get_state(self, data):
         # Define a transformation to resize the image
         transform = transforms.Resize((32, 32))
         # Apply the transformation to the input data
         image = transform(data)
         # Don't change view for CNN input.
        #  image = image.view(image.size(0), -1) # TODO Let's toggle this on and off...16 is so small, probably too smal for details.
         return image