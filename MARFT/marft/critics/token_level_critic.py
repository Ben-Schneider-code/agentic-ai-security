import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py


class TokenCritic(nn.Module):
    def __init__(
        self,
        model_path,
        device,
        bf16: bool = True,
        num_padding_at_beginning: int = 0,
        load_in_4bit: bool = False,
    ):
        super().__init__()

        self.device = device
        self.load_in_4bit = load_in_4bit

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            )
            device_map = {"": self.device}
        else:
            quantization_config = None
            # Load directly onto the target device (same pattern as Agent)
            # to avoid creating a CUDA context on GPU 0 during CPU→device transfer
            device_map = {"": self.device}

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            quantization_config=quantization_config,
            device_map=device_map,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.config = self.base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False).to(
                self.device
            )
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = (
                self.config.hidden_size
                if hasattr(self.config, "hidden_size")
                else self.config.n_embd
            )
            self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False).to(
                self.device
            )
            self.v_head_mlp2 = nn.Linear(1024, 512, bias=False).to(self.device)
            self.v_head_mlp3 = nn.Linear(512, 1, bias=False).to(self.device)
            self.relu = nn.ReLU()
        self.rwtransformer = self.base_model
        for param in self.rwtransformer.parameters():
            param.requires_grad = False
        self.PAD_ID = self.tokenizer.pad_token_id
        self.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False,
    ):
        with torch.no_grad():
            transformer_outputs = self.rwtransformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True,
            )

        hidden_states = transformer_outputs[1][-1].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x)
        return values

    def save_value_head(self, ckpt_path: str):
        """
        Persist only the value-head parameters.
        """
        # 1. Grab *this* module’s state-dict …
        full_sd = self.state_dict()
        # 2. … keep the keys that belong to the value head.
        vh_sd = {
            k: v
            for k, v in full_sd.items()
            if k.startswith("v_head") or k.startswith("v_head_mlp")
        }
        torch.save(vh_sd, ckpt_path)

    def load_value_head(self, ckpt_path: str, map_location="cpu"):
        """
        Restore the value head, leaving the transformer untouched.
        """
        vh_sd = torch.load(ckpt_path, map_location=map_location)
        # strict=False → ignore keys that aren’t in `vh_sd`
        missing, unexpected = self.load_state_dict(vh_sd, strict=False)
        missing = [k for k in missing if not k.startswith("rwtransformer")]
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            print(f"[Value head] missing keys (usually fine): {missing}")
