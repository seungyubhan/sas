import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        rtg_target = torch.clone(rtg[:,:-1])

        state_preds, action_preds, rtg_preds, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, deterministic=self.deterministic, 
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        state_dim = state_preds.shape[2]
        # state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_preds = torch.transpose(state_preds.reshape(-1, state_dim), 0, 1) * attention_mask.reshape(-1)
        state_preds = state_preds.reshape(self.batch_size, -1, state_dim)
        # state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = torch.transpose(state_target.reshape(-1, state_dim), 0, 1) * attention_mask.reshape(-1)
        state_target = state_target.reshape(self.batch_size, -1, state_dim)

        for i in range(state_preds.shape[0]):
            for j in range(state_preds.shape[1]):
                if state_preds[i][j] != np.zeros([state_dim]) and state_target[i][j] != np.zeros([state_dim]):
                    break
                if state_preds[i][j] == np.zeros([state_dim]) \
                        and state_target[i][j] == np.zeros([state_dim]) \
                        and state_preds[i][j+1] != np.zeros([state_dim]) \
                        and state_target[i][j+1] != np.zeros([state_dim]):
                    state_target[i][j+1] = 0

        rtg_dim = rtg_preds.shape[2]
        # rtg_preds = rtg_preds.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
        rtg_preds = torch.transpose(rtg_preds.reshape(-1, rtg_dim), 0, 1) * attention_mask.reshape(-1)
        rtg_preds = rtg_preds.reshape(self.batch_size, -1, rtg_dim)
        # rtg_target = rtg_target.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
        rtg_target = torch.transpose(rtg_target.reshape(-1, rtg_dim), 0, 1) * attention_mask.reshape(-1)
        rtg_target = rtg_target.reshape(self.batch_size, -1, rtg_dim)

        for i in range(rtg_preds.shape[0]):
            for j in range(rtg_preds.shape[1]):
                if rtg_preds[i][j] != 0 and rtg_target[i][j] != 0:
                    break
                if rtg_preds[i][j] == 0 and rtg_target[i][j] == 0 \
                        and rtg_preds[i][j+1] != 0 and rtg_target[i][j+1] != 0:
                    rtg_target[i][j+1] = 0

        loss = self.loss_fn(
            state_preds[:,:-1,:], action_preds, rtg_preds[:,:-1,:],
            state_target[:,1:,:], action_target, rtg_target[:,1:,:],
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = round(torch.mean((action_preds-action_target)**2).detach().cpu().item(), 5)
            self.diagnostics['training/state_error'] = round(torch.mean(0.1*(state_preds[:,:-1,:]-state_target[:,1:,:])**2).detach().cpu().item(), 5)
            self.diagnostics['training/rtg_error'] = round(torch.mean((rtg_preds[:,:-1,:]-rtg_target[:,1:,:])**2).detach().cpu().item(), 5)

        return loss.detach().cpu().item()
