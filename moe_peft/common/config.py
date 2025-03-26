import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypeAlias, Union

import torch

Tokens: TypeAlias = List[int]
Labels: TypeAlias = List[int]
Masks: TypeAlias = List[bool]
Ground: TypeAlias = List[str]
Citations: TypeAlias = List[str]
Query: TypeAlias = List[str]


@dataclass
class Prompt:
    instruction: str = None
    input: str = None
    label: str = None


@dataclass
class InputData:
    inputs: List[Union[Prompt, List[str], str]] = None
    prefix_length_: int = None
    tokens: Optional[Tokens] = None
    labels: Optional[Labels] = None
    grounds: Optional[Ground] = None
    citations: Optional[Citations] = None
    citation_tokens: Optional[List] = None
    citation_embeds: Optional[List] = None
    query: Optional[Query] = None
    token_len: Optional[int] = None
    prompt: Optional[str] = None
    prompt_len: Optional[int] = None
    test_citations: Optional[Citations] = None


@dataclass
class LLMModelConfig:
    name_or_path_: str = None
    device_: str = None
    dim_: int = None
    head_dim_: int = None
    intermediate_: int = None
    n_heads_: int = None
    n_kv_heads_: int = None
    n_layers_: int = None
    hidden_act_: str = None
    hidden_dropout_: float = None
    vocab_size_: int = None
    pad_token_id_: int = None
    rope_theta_: float = None
    partial_rotary_factor_: float = None
    max_seq_len_: int = None
    # eager or flash_attn
    attn_implementation_: str = "eager"
    # data type
    dtype_: torch.dtype = None


@dataclass
class LLMModelOutput:
    adapter_name: str = None
    logits: torch.Tensor = None
    router_logits: torch.Tensor = None
    loss: torch.Tensor = None
    cite_flag: bool = False
    aux_loss: torch.Tensor = None
    # for internal use
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    loss_fn_: Callable = None


@dataclass
class LLMBatchConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


def _efficient_operator_factory():
    efficient_operator = os.getenv("MOE_PEFT_EVALUATE_MODE") is None
    return efficient_operator


@dataclass
class LLMModelInput:
    batch_configs_: List[LLMBatchConfig] = None
    batch_tokens_: List[Tokens] = None
    batch_labels_: List[Labels] = None
    batch_grounds_: List[Ground] = None
    batch_cites: List[List] = None
    batch_cites_value: List[List] = None
    batch_masks_: List[Masks] = None
    batch_docs: List[str] = None
    batch_prompt_len: List[int] = None

    output_router_logits_: bool = True

    gradient_checkpoint_: str = "none"
    efficient_operator_: bool = field(default_factory=_efficient_operator_factory)
    inference_mode_: bool = False


@dataclass
class AdapterConfig:
    adapter_name: str = ""
    task_name: str = "casual"

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            adapter_name=config.get("name", None),
            task_name=config.get("task_name", None),
        )


lora_target_modules = {
    # LLaMA names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "dense": False,
    "fc1": False,
    "fc2": False,
    # Phi3 names
    "qkv_proj": False,
    "o_proj": False,
    "gate_up_proj": False,
    "down_proj": False,
    # GLM names
    "qkv_proj": False,
    "dense": False,
    "dense_h_to_4h": False,
    "dense_4h_to_h": False,
}


@dataclass
class LoraConfig(AdapterConfig):
    # Weight-Decomposed Low-Rank Adaptation
    use_dora_: bool = False
    # Rank-Stabilized LoRA
    # sets the adapter scaling factor to `alpha/math.sqrt(r)`
    use_rslora_: bool = False
    # can be original or gaussian
    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None
    atten_coin: float = None
    router_coin: float = None
    cite_coin:float = None
    learning_rate: float = None

    def check(self) -> "LoraConfig":
        assert isinstance(self.use_dora_, bool)
        assert isinstance(self.use_rslora_, bool)
        assert isinstance(self.lora_init_, str) and self.lora_init_ in [
            "original",
            "gaussian",
        ]
        assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_, float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, Dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.use_dora_ = config.get("use_dora", False)
        lora_config.use_rslora_ = config.get("use_rslora", False)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = config["r"]
        lora_config.lora_alpha_ = config["lora_alpha"]
        lora_config.lora_dropout_ = config["lora_dropout"]
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        lora_config.atten_coin = config["atten_mat_coin"]
        lora_config.router_coin = config["router_coin"]
        lora_config.cite_coin = config["cite_coin"]
        lora_config.learning_rate = config["lr"]
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = {}
        if self.use_dora_:
            config["use_dora"] = True
        if self.use_rslora_:
            config["use_rslora"] = True
        config["bias"] = "none"
        config["peft_type"] = "LORA"
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        config["atten_mat_coin"] = self.atten_coin
        config["router_coin"] = self.router_coin
        config["cite_coin"] = self.cite_coin
        config["lr"] = self.learning_rate

        return config
