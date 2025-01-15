import math
import time

from dqn_agent import *

import torch
from torch.nn import functional

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class TrainingTestingAgent():

    def __init__(self, buffer_batch_size, get_reward_every, batch_size, model, att_dim, n_patches, epochs, env, buffer_size, gamma, tau, update_target_every, lr, eps_end, eps_start, eps_decay, train_loader, validation_loader, device, dqn_input_size, pretrained=False):

      self.env = env

      self.epochs = epochs
      self.eps_start = eps_start
      self.eps_end = eps_end
      self.eps_decay = eps_decay
      self.get_reward_every = get_reward_every

      self.batch_size = batch_size

      self.validation_acc = []
      self.validation_loss = []
      self.epoch_time_list = []
      self.validation_precision = []
      self.validation_recall = []
      self.validation_f1 = []

      self.ViTnet = model

      self.n_patches = n_patches

      # creazione agente
      self.agent = DQNAgent(buffer_batch_size, att_dim, n_patches, buffer_size, gamma, tau, update_target_every, lr, self.env, device, dqn_input_size)

      self.train_loader = train_loader
      self.validation_loader = validation_loader

      self.device = device


    def train_test(self, models_save_path, dataset_name):

      # lista dei reward ad ogni step
      step_reward = []            # 
      selected_patch_list = []    # A record of the agent-selected patches during the current
      epoch = 0                   # Training epoch (how many times have we run through the dataset?)
      iteration = 0               # Training iteration (how many dataset batches have we run?)
      eps = self.eps_start        # starting epsilon.

      while epoch < self.epochs:

          epoch += 1
          print(f'Epoch: {epoch}/{self.epochs}')
          self.env.current_epoch += 1

          start_time = time.time()
          samples = len(self.train_loader.dataset)

          for i, (data, target) in enumerate(self.train_loader):
              i +=1

              iteration += 1

              data = data.to(self.device)
              target = target.to(self.device)

              state = self.env.get_state(data)
              # print(state.shape())

              patch_list = self.agent.select_action(state, eps)

              selected_patch_list.append(patch_list)

              # alleno
              if i % self.get_reward_every != 0:
                self.env.step_train(patch_list, data, target)

              # alleno e testo il ViT
              else:
                new_state, reward = self.env.step_reward(patch_list, data, target)

                print(f'  Epsilon: {eps},   Reward: {reward}')

                step_reward.append(reward)

                reward = torch.full((self.batch_size, ), reward, dtype=torch.float32)

                if epoch != 1:
                  # inseriamo le osservazioni nella replayMemory
                  self.agent.memory.push(state, new_state, reward)

                  # ottimizziamo l'agente
                  self.agent.optimize_model()

              eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * iteration / self.eps_decay)

          print(f'\nInizio Testing')
          loss, acc, precision, recall, f1 = self.evaluate_epoch(self.validation_loader, models_save_path, dataset_name, self.device)

          epoch_time = time.time()-start_time

          self.validation_acc.append(acc)
          self.validation_loss.append(loss)
          self.validation_precision.append(precision)
          self.validation_recall.append(recall)
          self.validation_f1.append(f1)
          self.epoch_time_list.append(epoch_time)

          print(f'Epoch time: {epoch_time}')
          print("#"*40)


          print("#"*40)
          print('Episode End')
          print("#"*40)
          print("#"*40)

      return step_reward, selected_patch_list



    def evaluate_epoch(self, data_load, models_save_path, dataset_name, device):
        patches = self.ViTnet.get_patches()
        self.ViTnet.set_patches(torch.tensor([1 for i in range(len(patches))], dtype=torch.float))

        self.ViTnet.eval()


        elements = 0
        # predizioni corrette
        csamp = 0
        tloss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_load:

                elements += len(data)
                data = data.to(device)
                target = target.to(device)

                predictions = self.ViTnet(data)

                output = functional.log_softmax(predictions, dim=1)
                loss = functional.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)

                predictions = torch.argmax(predictions, dim=1).cpu().numpy()

                tloss += loss.item()
                csamp += pred.eq(target).sum()

                all_predictions.extend(predictions)
                all_targets.extend(target.cpu())

        loss_val = tloss / elements
        acc_val = (100.0 * csamp / elements).cpu()


        print('\n\nAverage test loss: ' + '{:.4f}'.format(loss_val) +
              '  Accuracy:' + '{:5}'.format(csamp) + '/' +
              '{:5}'.format(elements) + ' (' +
              '{:4.2f}'.format(acc_val) + '%)\n')


        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        return loss_val, acc_val, precision, recall, f1


    def train_info(self):

        return {
                'train_loss': self.env.train_loss_list,
                'train_time': self.env.train_time_list,
                }

    def validation_info(self):
        return {
                'validation_loss': self.validation_loss,
                'validation_acc': [tensor.item() for tensor in self.validation_acc],
                'validation_precision': self.validation_precision,
                'validation_recall': self.validation_recall,
                'validation_f1': self.validation_f1,
                'epoch_time': self.epoch_time_list
                }
    
def evaluate_agent(agent, data_load, device, vitnet, model, mode = False):

    agent.eval()

    elements = 0
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:

            elements += len(data)
            data = data.to(device)
            target = target.to(device)

            state = env.get_state(data)

            if mode == "agent":
              action = model.agent.select_action(state)
            elif mode == "random":
              action = model.env.action_space.sample()
            else:
              action = torch.tensor([1 for i in range(len(model.env.action_space.sample()))], dtype = torch.float)

            vitnet.set_patches(action)

            output = functional.log_softmax(agent(data), dim=1)
            loss = functional.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()

    loss_val = tloss / elements
    acc_val = (100.0 * csamp / elements).cpu()

    print('\nAverage validation loss: ' + '{:.4f}'.format(loss_val) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(elements) + ' (' +
          '{:4.2f}'.format(acc_val) + '%)\n')

    return loss_val, acc_val
