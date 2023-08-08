import pyrallis
import torch
import transformers
from torch import nn

import config


class RndModel(nn.Module):
    def __init__(
        self,
        llama_config: transformers.LlamaConfig,
        hidden_layers: int,
        emb_size: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llama = transformers.LlamaForCausalLM(llama_config)
        hidden = llama_config.hidden_size
        prior_layers = [nn.Linear(hidden, hidden) for _ in range(hidden_layers)] + [
            nn.Linear(hidden, emb_size)
        ]
        self.prior = nn.Sequential(*prior_layers)


@pyrallis.wrap()
def main(cfg: config.RndConfig):
    pass


if __name__ == "__main__":
    main()
