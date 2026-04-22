import numpy as np
import torch
import torch.nn as nn
import logging
import warnings
import inspect
from copy import deepcopy
import transformers

import logging
import tempfile
import os
import pickle



from torch.distributions.normal import Normal
import torch.nn.functional as F
from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

from enum import Enum
logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)



class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    GREEDY_SEARCH_WITH_OM = "greedy_search_with_om"
    GREEDY_SEARCH_WITH_OM_NO_PADDING = "greedy_search_with_om_no_padding"
    GREEDY_SEARCH_WITH_TRUE_CLF = "greedy_search_with_true_clf"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationMixin:
    def prepare_inputs_for_genearation(self, input_ids, **model_kwargs):
            # forward_args = inspect.getargspec(self.forward).args
            # states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
            if not isinstance(input_ids, dict):
                raise TypeError
            else:
                pass

            # ii = deepcopy(input_ids)                  

            # if self.max_length is not None:
            #     for k, v in ii.items():
            #         ii[k] = v[:,-self.max_length:]

            states = input_ids['states'][:,-self.max_length:].detach()
            actions = input_ids['actions'][:,-self.max_length:].detach()
            returns_to_go = input_ids['returns_to_go'][:,-self.max_length:].detach()
            
            timesteps = torch.arange(0, states.shape[1]).to(device=states.device).reshape(1,-1)
            # ii.update(timesteps=timesteps)

            attention_mask = torch.cat([torch.zeros(max(0,self.max_length-states.shape[1])), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            # states = ii['states']
            # actions = ii['actions']
            # returns_to_go = ii['returns_to_go']
            # timesteps = ii['timesteps']
            
            
            states = torch.cat(
                [torch.zeros((states.shape[0], max(0,self.max_length-states.shape[1]), self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max(0,self.max_length - actions.shape[1]), self.act_dim),
                                device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max(0,self.max_length-returns_to_go.shape[1]), 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max(0,self.max_length-timesteps.shape[1])), device=timesteps.device), timesteps],
                dim=1
                ).to(dtype=torch.long)
            rewards = None

            out_dict =dict(states=states,
                           actions=actions,
                           returns_to_go=returns_to_go,
                           timesteps=timesteps, rewards=rewards)

            # input_ids.update(states=states)
            # input_ids.update(actions=actions)
            # input_ids.update(returns_to_go=returns_to_go)
            # input_ids.update(timesteps=timesteps)
            # input_ids.update(rewards=rewards)
            return out_dict, attention_mask

    def _get_stopping_criteria(self, generation_config, stopping_criteria):
        # len1 = 
        # if len1 is None:
        #     sc = stopping_criteria
        # else:
        #     sc = min(len1, stopping_criteria)
        sc = stopping_criteria
        def stop_func(input_ids):
            seq_len = input_ids['states'].shape[1]
            return seq_len >= sc + generation_config
        return stop_func

    @torch.no_grad()
    def generate(self,
                 inputs,
                 generation_config=None,
                 stopping_criteria=None,
                 **kwargs):
        # states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        self.generation_config = generation_config
        model_kwargs = kwargs
        if stopping_criteria == None:
            stopping_criteria = self.max_ep_len


        # 2. set logit and stopping criteria

        # 3. define model inputs
        # input_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        #     inputs, generation_config)
        
        # 4. Define model kwargs

        attention_mask = None
        # attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
        # attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

        # 5. input id?
        input_ids = inputs
        prompt_len = input_ids["states"].size(1) 
        #inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare max_length depending on stopping criteria.
        # input_ids_length = input_ids.shape[-1]

        # 7. determine gernation mode
        if generation_config==None:
            generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = generation_config["output_mode"]

        # 8. prepare distribution pre_processing samplers

        # 9. prepare stopping crieria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=prompt_len, stopping_criteria=stopping_criteria
        )

        # 10. go into generation mode
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            return self.greedy_search(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.GREEDY_SEARCH_WITH_OM:
            return self.greedy_search_for_ocuppancy_measure(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.GREEDY_SEARCH_WITH_OM_NO_PADDING:
            return self.greedy_search_for_om_no_padding(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,)
        elif generation_mode == GenerationMode.GREEDY_SEARCH_WITH_TRUE_CLF:
            return self.greedy_search_for_clf(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,)
        else:
            raise ValueError
        
    @torch.no_grad()    
    def greedy_search_for_ocuppancy_measure(self,
                      input_ids,
                      stopping_criteria,
                      **model_kwargs):
        
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        om_arr = []
        while True:
            model_inputs, attention_mask = self.prepare_inputs_for_genearation(input_ids, **model_kwargs)
            # del a
            # states, actions, rewards, returns_to_go, timesteps,
        

            # forward pass to get next token
            # outputs = self(
            #     **model_inputs,
            #     attention_mask=attention_mask
            # )
            # print(model_inputs['states'][:,-20:])
            outputs = self.forward_with_occupancy_measure(**model_inputs, 
                                deterministic=True, 
                                max_k=model_kwargs['max_k'],
                                attention_masks=attention_mask)
            # outputs = self.forward(**model_inputs, deterministic=False, attention_mask=attention_mask)
            out_dicts = {'states': outputs[0].detach(),
                         'actions': outputs[1].detach(),
                         'returns_to_go': outputs[2].detach()} #, 'log_p_pi': outputs[3]}
            om_arr.append(outputs[3].detach())
            # del model_inputs, outputs
            # outputs: state_preds, action_preds, return_preds

            # next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # store scores, atttenstions and hidden states when required?
            
            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            # print(out_dicts['states'].size(), input_ids['states'].size())

            # output_ids = {k: torch.cat([v, torch.unsqueeze(v[:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            input_ids.update({k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()})

            # model_kwargs = self._update_kwargs_for_generation(
            #     outputs, model_kwargs
            # )

            # stop if we exceed the max length
            if stopping_criteria(input_ids):
                this_peer_finished = True
                break
            torch.cuda.empty_cache()
        input_ids.update(occupancy_measure=torch.cat(om_arr, dim=0).unsqueeze(0))

        return input_ids

    @torch.no_grad()    
    def greedy_search_for_om_no_padding(self,
                input_ids,
                    stopping_criteria,
                    **model_kwargs):
        
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        om_arr = []
        while True:
            model_inputs, attention_mask = self.prepare_inputs_for_genearation(input_ids, **model_kwargs)
            # del a
            # states, actions, rewards, returns_to_go, timesteps,
        

            # forward pass to get next token
            # outputs = self(
            #     **model_inputs,
            #     attention_mask=attention_mask
            # )
            # print(model_inputs['states'][:,-20:])
            outputs = self.forward_with_om_no_padding(**model_inputs, 
                                deterministic=True, 
                                max_k=model_kwargs['max_k'],
                                attention_masks=attention_mask)
            # outputs = self.forward(**model_inputs, deterministic=False, attention_mask=attention_mask)
            out_dicts = {'states': outputs[0].detach(),
                         'actions': outputs[1].detach(),
                         'returns_to_go': outputs[2].detach()} #, 'log_p_pi': outputs[3]}
            om_arr.append(outputs[3].detach())
            # del model_inputs, outputs
            # outputs: state_preds, action_preds, return_preds

            # next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # store scores, atttenstions and hidden states when required?
            
            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            # print(out_dicts['states'].size(), input_ids['states'].size())

            # output_ids = {k: torch.cat([v, torch.unsqueeze(v[:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            input_ids.update({k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()})

            # model_kwargs = self._update_kwargs_for_generation(
            #     outputs, model_kwargs
            # )

            # stop if we exceed the max length
            if stopping_criteria(input_ids):
                this_peer_finished = True
                break
            torch.cuda.empty_cache()
        input_ids.update(occupancy_measure=torch.cat(om_arr, dim=0).unsqueeze(0))

        return input_ids
    
    @torch.no_grad()    
    def greedy_search_for_clf(self,
                input_ids,
                    stopping_criteria,
                    **model_kwargs):
        
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        om_arr = []
        previous_om = torch.tensor([float('inf')], device='cuda')
        satisfy_condition = torch.zeros(0, device='cuda')
        while True:
            model_inputs, attention_mask = self.prepare_inputs_for_genearation(input_ids, **model_kwargs)
            # del a
            # states, actions, rewards, returns_to_go, timesteps,
        

            # forward pass to get next token
            # outputs = self(
            #     **model_inputs,
            #     attention_mask=attention_mask
            # )
            # print(model_inputs['states'][:,-20:])
            outputs = self.forward_with_om_no_padding(**model_inputs, 
                                deterministic=True, 
                                max_k=model_kwargs['max_k'],
                                attention_masks=attention_mask)
            # outputs = self.forward(**model_inputs, deterministic=False, attention_mask=attention_mask)
            out_dicts = {'states': outputs[0].detach(),
                         'actions': outputs[1].detach(),
                         'returns_to_go': outputs[2].detach()} #, 'log_p_pi': outputs[3]}
            om = outputs[3].detach()
            om_arr.append(om)
            satisfy_condition = torch.cat((satisfy_condition, torch.le(om, previous_om)), 0)
            previous_om = om

            # del model_inputs, outputs
            # outputs: state_preds, action_preds, return_preds

            # next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # store scores, atttenstions and hidden states when required?
            
            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            # print(out_dicts['states'].size(), input_ids['states'].size())

            # output_ids = {k: torch.cat([v, torch.unsqueeze(v[:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            input_ids.update({k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()})

            # model_kwargs = self._update_kwargs_for_generation(
            #     outputs, model_kwargs
            # )

            # stop if we exceed the max length
            if stopping_criteria(input_ids):
                this_peer_finished = True
                break
            torch.cuda.empty_cache()
        input_ids.update(occupancy_measure=torch.cat(om_arr, dim=0).unsqueeze(0))
        input_ids.update(satisfy_condition=torch.sum(satisfy_condition))


        return input_ids
        
    @torch.no_grad()    
    def greedy_search(self,
                      input_ids,
                      stopping_criteria,
                      **model_kwargs):
        
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        logp_p_pi_arr = []
        while True:
            model_inputs, attention_mask = self.prepare_inputs_for_genearation(input_ids, **model_kwargs)
            # states, actions, rewards, returns_to_go, timesteps,
        

            # forward pass to get next token
            # outputs = self(
            #     **model_inputs,
            #     attention_mask=attention_mask
            # )
            # print(model_inputs['states'][:,-20:])
            outputs = self.forward(**model_inputs, deterministic=False, attention_mask=attention_mask)
            out_dicts = {'states': outputs[0].detach(),
                         'actions': outputs[1].detach(),
                         'returns_to_go': outputs[2].detach()} #, 'log_p_pi': outputs[3]}
            logp_p_pi_arr.append(outputs[3].detach())
            # output_ids = {k: torch.cat([v, torch.unsqueeze(v[:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            # input_ids.update(output_ids)
            input_ids.update({k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()})
            # del model_inputs, attention_mask, outputs
            # outputs: state_preds, action_preds, return_preds

            # next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # store scores, atttenstions and hidden states when required?
            
            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            # print(out_dicts['states'].size(), input_ids['states'].size())
            # input_ids = {k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()}
            # output_ids = {k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            # model_kwargs = self._update_kwargs_for_generation(
            #     outputs, model_kwargs
            # )

            # stop if we exceed the max length
            if stopping_criteria(input_ids):
                this_peer_finished = True
                break
            torch.cuda.empty_cache()
        input_ids.update(log_p_pi=torch.cat(logp_p_pi_arr, dim=0).unsqueeze(0))

        return input_ids
    
  
    @torch.no_grad()
    def run_single_step(self,
                 inputs,
                 generation_config=None,
                 stopping_criteria=None,
                 **kwargs):
        # states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        self.generation_config = generation_config
        model_kwargs = kwargs
        if stopping_criteria == None:
            stopping_criteria = 100


        # 2. set logit and stopping criteria

        # 3. define model inputs
        # input_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        #     inputs, generation_config)
        
        # 4. Define model kwargs

        attention_mask = None
        # attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
        # attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

        # 5. input id?
        input_ids = inputs
        prompt_len = input_ids["states"].size(1) 
        #inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare max_length depending on stopping criteria.
        # input_ids_length = input_ids.shape[-1]

        # 7. determine gernation mode
        generation_mode = GenerationMode.GREEDY_SEARCH

        # 8. prepare distribution pre_processing samplers

        # 9. prepare stopping crieria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=prompt_len, stopping_criteria=1
        )

        # 10. go into generation mode
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            return self.greedy_search(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,
            )
        else:
            raise ValueError
        
class DecisionTransformer(TrajectoryModel, GenerationMixin):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            act_limit,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.max_ep_len = max_ep_len
        self.act_limit = act_limit

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state_net = torch.nn.Linear(hidden_size, hidden_size)
        self.predict_state_mu = torch.nn.Linear(hidden_size, state_dim)
        self.predict_state_logstd = nn.Linear(hidden_size, state_dim)

        # self.predict_action_net = nn.Sequential(
        #     *([nn.Linear(hidden_size, hidden_size)] + ([nn.Tanh()] if action_tanh else []))
        # )      
        self.predict_action_mu = nn.Linear(hidden_size, act_dim)
        self.predict_action_logstd = nn.Linear(hidden_size, act_dim)

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, deterministic, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # state_net_out = self.predict_state_net(x[:,2])    # predict next state given state and action
        state_mu = self.predict_state_mu(x[:,2])
        state_logstd = self.predict_state_logstd(x[:,2])
        state_logstd = torch.clamp(state_logstd, -20, 2)
        state_std = torch.exp(state_logstd)
        state_distribution = Normal(state_mu, state_std)
        if deterministic:
            state_preds = state_mu
        else:
            state_preds = state_distribution.rsample()
        log_p = state_distribution.log_prob(state_preds).sum(axis=-1)
        log_p -= (2*(np.log(2) - state_preds - F.softplus(-2*state_preds))).sum(axis=-1)

        # action_net_out = self.predict_action_net(x[:,1])
        action_mu = self.predict_action_mu(x[:,1])
        action_logstd = self.predict_action_logstd(x[:,1])
        action_logstd = torch.clamp(action_logstd, -20, 2)
        action_std = torch.exp(action_logstd)
        action_distribution = Normal(action_mu, action_std)
        if deterministic:
            pi_action = action_mu
        else:
            pi_action = action_distribution.rsample()  # predict next action given state
        log_pi = action_distribution.log_prob(pi_action).sum(axis=-1)
        log_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)

        # action scale
        action_preds = torch.tanh(pi_action)
        action_preds = torch.from_numpy(self.act_limit[0]).to(device=actions.device) \
            + torch.from_numpy(self.act_limit[1]).to(device=actions.device) * action_preds

        return_preds = self.predict_return(x[:,2])  # predict next return given state and action

        # log_p_pi = state_logstd + action_logstd
        log_p_pi = log_p[:, -2:-1] + log_pi[:, -1:]

        return state_preds, action_preds, return_preds, log_p_pi

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, deterministic, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, deterministic, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


    def forward_with_occupancy_measure(self, states, actions, rewards, returns_to_go, timesteps, deterministic, max_k, attention_masks=None, **kwargs):
        states_arr = []
        a_arr = []
        rtg_arr = []
        t_arr = []
        mask_arr = []
        window_length = states.size(1)

        # pad all tokens to sequence length
        for k_ in range(max_k):
            # k_len = k_ + 1
            k_len = window_length - k_

            states_ = states[:,-k_len:]
            actions_ = actions[:,-k_len:]
            returns_to_go_ = returns_to_go[:,-k_len:]
            timesteps_ = timesteps[:,-k_len:]

            attention_mask_ = torch.cat([torch.zeros(max(0,window_length-states_.shape[1])), torch.ones(states_.shape[1])])
            attention_mask_ = attention_mask_.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states_ = torch.cat(
                [torch.zeros((states.shape[0], max(0,window_length-states_.shape[1]), self.state_dim), device=states.device), states_],
                dim=1).to(dtype=torch.float32)
            
            actions_ = torch.cat(
                [torch.zeros((actions.shape[0], max(0,window_length - actions_.shape[1]), self.act_dim),
                                device=actions.device), actions_],
                dim=1).to(dtype=torch.float32)
            
            returns_to_go_ = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max(0,window_length-returns_to_go_.shape[1]), 1), device=returns_to_go.device), returns_to_go_],
                dim=1).to(dtype=torch.float32)
            
            timesteps_ = torch.cat(
                [torch.zeros((timesteps.shape[0], max(0,window_length-timesteps_.shape[1])), device=timesteps.device), timesteps_],
                dim=1
            ).to(dtype=torch.long)

            states_arr.append(states_)
            a_arr.append(actions_)
            rtg_arr.append(returns_to_go_)
            t_arr.append(timesteps_)
            mask_arr.append(attention_mask_)

        # states_arr.append(states)
        # a_arr.append(actions)
        # rtg_arr.append(returns_to_go)
        # t_arr.append(timesteps)
        # mask_arr.append(attention_masks)
        
        states_cat = torch.cat(states_arr,dim=0)
        actions_cat = torch.cat(a_arr, dim=0)
        returns_to_go_cat = torch.cat(rtg_arr, dim=0)
        timesteps_cat = torch.cat(t_arr, dim=0)
        attention_mask_cat = torch.cat(mask_arr, dim=0)

        state_preds, action_preds, return_preds, log_p_pi = self.forward(
            states_cat, actions_cat, None, returns_to_go_cat, timesteps_cat, deterministic, attention_mask=attention_mask_cat, **kwargs)
        
        gamma = 0.99
        # gamma_vector = torch.pow(gamma, torch.flip(torch.arange(max_k), dims=(0,))).cuda()
        gamma_vector = torch.pow(gamma, torch.flip(torch.arange(window_length - max_k, window_length), dims=(0,))).cuda()
        occupancy_measure = torch.tensordot(torch.exp(log_p_pi), gamma_vector, dims=([0], [0]))
        
        return state_preds[:1,...], action_preds[:1,...], return_preds[:1,...], occupancy_measure
    
    def forward_with_om_no_padding(self, states, actions, rewards, returns_to_go, timesteps, deterministic, max_k, attention_masks=None, **kwargs):
        
        
        window_length = states.size(1)
        states_pred = []
        actions_pred = []
        rtgs_pred = []
        log_p_pis = torch.zeros((max_k, 1), device=states.device)

        # pad all tokens to sequence length
        for k_ in range(max_k):
            # k_len = k_ + 1
            k_len = window_length - k_

            states_ = states[:,-k_len:]
            actions_ = actions[:,-k_len:]
            returns_to_go_ = returns_to_go[:,-k_len:]
            timesteps_ = timesteps[:,-k_len:]
            attention_mask_ = torch.ones(1, states_.shape[1]).to(dtype=torch.long, device=states.device)

            state_pred, action_pred, rtg_pred, log_p_pi = self.forward(
            states_, actions_, None, returns_to_go_, timesteps_, deterministic, attention_mask=attention_mask_, **kwargs)

            states_pred.append(state_pred)
            actions_pred.append(action_pred)
            rtgs_pred.append(rtg_pred)
            log_p_pis[k_] = log_p_pi
                
        gamma = 0.99
        # gamma_vector = torch.pow(gamma, torch.flip(torch.arange(max_k), dims=(0,))).cuda()
        gamma_vector = torch.pow(gamma, torch.flip(torch.arange(window_length - max_k, window_length), dims=(0,))).to(device=states.device)
        occupancy_measure = torch.tensordot(torch.exp(log_p_pis), gamma_vector, dims=([0], [0]))
        
        return states_pred[0], actions_pred[0], rtgs_pred[0], occupancy_measure
