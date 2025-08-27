import torch
from pydantic import BaseModel
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch import nn as nn

from components.episode_buffer import EpisodeBatch


class CoreConfig(BaseModel):
    num_attention_heads: int = 4
    core_hidden_size: int = 64
    mem: bool = True
    max_position_embeddings: int = 16384
    add_cross_attention: bool = True


class TransformerCore(nn.Module):
    def __init__(self, cfg, input_size: int):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.core_cfg: CoreConfig = CoreConfig()
        self.use_memory = cfg.core_memory
        self.use_global_memory = cfg.use_global_memory
        self.num_agents = cfg.n_agents
        self.remember_time_steps = 8
        core_cfg_copy = CoreConfig().__dict__
        core_cfg_copy['hidden_size'] = core_cfg_copy.pop('core_hidden_size')

        self.encoder = nn.Linear(input_size, self.core_cfg.core_hidden_size)
        self.core_transformer = GPT2Block(GPT2Config(**core_cfg_copy))
        self.wpe = nn.Embedding(core_cfg_copy['max_position_embeddings'],
                                self.core_cfg.core_hidden_size)
        self.decoder = nn.Linear(self.core_cfg.core_hidden_size, input_size)
        if self.use_memory:
            self.mem_head = nn.Linear(self.core_cfg.core_hidden_size,
                                      self.core_cfg.core_hidden_size,
                                      bias=False)

        self.ln_f = nn.LayerNorm(self.core_cfg.core_hidden_size, eps=1e-5)

        self.global_memory = None
        self.agent_memory = None
        self.history_seq = None
        self.init_mem()

        self.to(self.cfg.device)


    def init_mem(self):
        self.global_memory = torch.zeros((self.num_agents, self.core_cfg.core_hidden_size),
                                         device=self.cfg.device) if getattr(self.cfg, 'use_global_memory', False) else None
        self.agent_memory = torch.zeros((self.num_agents, self.core_cfg.core_hidden_size),
                                        device=self.cfg.device) if getattr(self.cfg, 'core_memory', False) else None
        self.history_seq = torch.zeros((self.num_agents, self.remember_time_steps, self.core_cfg.core_hidden_size),
                                       device=self.cfg.device)

    def forward(self, core_input):
        is_seq = not torch.is_tensor(core_input)
        if not is_seq:
            core_input = core_input.unsqueeze(1)
            first_time_mem = False
            if self.use_memory:
                # first pass with empty memory
                if not self.agent_memory.abs().sum().is_nonzero():
                    self.agent_memory = None
                    first_time_mem = True
                else:
                    agent_memory_batch = self.agent_memory.unsqueeze(1)
                    restored_global_memory = self.global_memory.repeat(5, 1, 1)
        if self.history_seq is not None:
            inputs = torch.cat([self.history_seq, core_input], dim=1)
        else:
            inputs = core_input.contiguous()

        if self.agent_memory is not None:
            inputs = torch.cat([agent_memory_batch, inputs], dim=1)

        position_ids = torch.arange(0, inputs.size(1), dtype=torch.long).to('cuda')
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs + position_embeds

        encoder_hidden_states = None
        if self.agent_memory is not None:
            if self.use_global_memory:
                encoder_hidden_states = restored_global_memory.contiguous()
        x = self.core_transformer(hidden_states=hidden_states.contiguous(),
                                  encoder_hidden_states=encoder_hidden_states,
                                  )[0]
        x = self.ln_f(x)
        core_out = x[:, -1:]

        if self.use_memory:
            if first_time_mem:
                my_new_mem = core_out.contiguous()
            else:
                my_new_mem, _ = torch.split(x, [1, x.size()[1] - 1], dim=1)
            my_new_mem = self.mem_head(my_new_mem)

        # update history with current head_output
        if self.history_seq is not None:
            self.history_seq = torch.cat([self.history_seq[:, 1:], core_input], dim=1)

        if not is_seq:
            core_out = core_out.squeeze(1)
            if self.use_memory:
                self.agent_memory = my_new_mem.squeeze(1)

        if self.global_memory is not None:
            self.global_memory = self.agent_memory


        return core_out

    def forward_batch(self, batch: EpisodeBatch):
        bs, max_t = batch.batch_size, batch.max_seq_length
        obs = batch['obs']
        processed_obs = torch.zeros_like(obs)

        for b in range(bs):
            self.init_mem()
            for seq in range(max_t):
                data = obs[b, seq, :, :]
                enc_out = self.encoder(data)
                core_out = self.forward(enc_out)
                processed_obs[b, seq, :, :] = self.decoder(core_out)

        return processed_obs

    def __call__(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)

    def get_out_size(self) -> int:
        return self.core_cfg.core_hidden_size
