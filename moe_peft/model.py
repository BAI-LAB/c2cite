import copy
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from moe_peft.adapters import (
    LoraMoeConfig,
    MixLoraConfig,
    MolaConfig,
    lora_config_factory,
    moe_layer_factory,
    router_loss_factory,
)
from moe_peft.common import (
    CHECKPOINT_CLASSES,
    AdapterConfig,
    Linear,
    LLMCache,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    LLMModelOutput,
    LLMMoeBlock,
    LLMOutput,
    LoraConfig,
    unpack_router_logits,
)
from moe_peft.executors import executor
from moe_peft.models import from_pretrained
from moe_peft.tasks import SequenceClassificationTask, task_dict
from moe_peft.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from moe_peft.utils import BitsAndBytesConfig


class CasualOutputLayer(LLMOutput):
    def __init__(self, vocab_size: int, weight: torch.nn.Linear):
        super().__init__()
        self.vocab_size_: int = vocab_size
        self.lm_head_: torch.nn.Module = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels, 
        cites: Optional[List] = None, cites_v: Optional[List] = None, prompt_lens: Optional[List] = None
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=torch.long, device=output_logits.device)
            )
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=output_logits.device)
        
        
        loss_fn = torch.nn.CrossEntropyLoss()    
        if cites:
            for i in range(len(labels)):
                for j in range(len(cites_v[i])):
                    labels[i][cites[i][j]] = -100
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        

        """return loss_fn(
            output_logits[..., :-1, :].contiguous().view(-1, self.vocab_size_),
            labels[..., 1:].contiguous().view(-1),
        )"""
        ans = 0
        for i in range(len(prompt_lens)):
            ans += loss_fn(
                output_logits[i, prompt_lens[i] - 1:-1, :].contiguous().view(-1, self.vocab_size_),
                labels[i, prompt_lens[i]:].contiguous().view(-1),
                )
        return ans / len(prompt_lens)


class ClassificationOutputLayer(LLMOutput):
    def __init__(
        self,
        task_type: str,
        num_labels: int,
        label_dtype: torch.dtype,
        hidden_size: int,
        pad_token_id: int,
        device: str,
        weight: Optional[torch.Tensor],
    ):
        super().__init__()
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type
        self.pad_id_ = pad_token_id
        self.score_ = torch.nn.Linear(
            hidden_size,
            self.num_labels_,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        if weight is None:
            torch.nn.init.kaiming_normal_(self.score_.weight, a=math.sqrt(5))
        else:
            with torch.no_grad():
                self.score_.weight.copy_(weight["classifier"])

    def state_dict(self):
        return {"classifier": self.score_.weight}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.score_(data.to(torch.float32))

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=self.label_dtype_, device=output_logits.device)
            )
        else:
            labels = torch.tensor(
                labels, dtype=self.label_dtype_, device=output_logits.device
            )
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(
            output_logits.device
        )
        pooled_logits = output_logits[
            torch.arange(batch_size, device=output_logits.device), sequence_lengths
        ]
        if self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_: Dict[str, torch.nn.Module] = {}

    def forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> List[LLMModelOutput]:
        outputs = []
        for lora_config in input_args.batch_configs_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            assert adapter_name != "" and adapter_name in self.layers_
            layer = self.layers_[adapter_name]
            outputs.append(
                LLMModelOutput(
                    adapter_name=adapter_name,
                    logits=layer.forward(data[start_idx:end_idx]),
                    loss_fn_=layer.loss,
                )
            )

        return outputs


def init_lora_layer_weight(
    transformer_layer: LLMDecoder,
    llm_config: LLMModelConfig,
    lora_config: LoraConfig,
    lora_weights: Optional[Dict[str, torch.Tensor]],
):
    target_modules = lora_config.target_modules_
    attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
    attn_state_dict: Dict[str, torch.Tensor]
    mlp_state_dict: Dict[str, torch.Tensor]
    all_state_dict: Dict[str, torch.Tensor] = copy.copy(attn_state_dict)
    all_state_dict.update(mlp_state_dict)
    moe_init_strategy = "none"
    if isinstance(lora_config, MixLoraConfig):
        model_prefix_name = "mixlora"
        moe_layer_name_list = list(mlp_state_dict.keys())
        moe_init_strategy = "fused_mlp"
    elif isinstance(lora_config, LoraMoeConfig):
        model_prefix_name = "loramoe"
        moe_layer_name_list = list(mlp_state_dict.keys())
        moe_init_strategy = "plugin"
    elif isinstance(lora_config, MolaConfig):
        model_prefix_name = "mola"
        moe_layer_name_list = list(all_state_dict.keys())
        moe_init_strategy = "plugin"
    else:
        model_prefix_name = "base_model.model.model"
        moe_layer_name_list = []

    assert len(moe_layer_name_list) == 0 or moe_init_strategy in ["plugin", "fused_mlp"]

    if moe_init_strategy == "fused_mlp":
        transformer_layer.mlp_.moes_[lora_config.adapter_name] = moe_layer_factory(
            llm_config.dim_,
            llm_config.device_,
            lora_config,
            (
                None
                if lora_weights is None
                else lora_weights[
                    f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.mlp.moe_gate.weight"
                ]
            ),
        )

    for proj_name, lora_linear in all_state_dict.items():
        lora_linear: Linear
        if proj_name not in target_modules or not target_modules[proj_name]:
            continue
        module_name = (
            "self_attn"
            if proj_name in attn_state_dict
            else ("mlp" if proj_name in mlp_state_dict else None)
        )
        module_name = f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.{module_name}.{proj_name}"
        if proj_name in moe_layer_name_list:
            if moe_init_strategy == "plugin":
                # init for gating mechanisms
                lora_linear.moes_[lora_config.adapter_name] = moe_layer_factory(
                    lora_linear.in_features_,
                    llm_config.device_,
                    lora_config,
                    (
                        lora_weights.get(f"{module_name}.moe_gate.weight", None)
                        if lora_weights is not None
                        else None
                    ),
                )

            for expert_idx in range(lora_config.num_experts_):
                if lora_weights is None:
                    lora_a = None
                    lora_b = None
                else:
                    lora_a = lora_weights.get(
                        f"{module_name}.experts.{expert_idx}.lora_A.weight", None
                    )
                    lora_b = lora_weights.get(
                        f"{module_name}.experts.{expert_idx}.lora_B.weight", None
                    )

                lora_linear.init_lora_weight(
                    lora_config.expert_config(expert_idx), (lora_a, lora_b)
                )
        else:
            if lora_weights is None:
                lora_a = None
                lora_b = None
            else:
                lora_a = lora_weights.get(f"{module_name}.lora_A.weight", None)
                lora_b = lora_weights.get(f"{module_name}.lora_B.weight", None)

            lora_linear.init_lora_weight(lora_config, (lora_a, lora_b))


def get_lora_layer_weight(
    transformer_layer: LLMDecoder,
    lora_config: LoraConfig,
    lora_weights: Dict[str, torch.Tensor],
):
    target_modules = lora_config.target_modules_
    attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
    attn_state_dict: Dict[str, torch.Tensor]
    mlp_state_dict: Dict[str, torch.Tensor]
    all_state_dict: Dict[str, torch.Tensor] = copy.copy(attn_state_dict)
    all_state_dict.update(mlp_state_dict)
    if isinstance(lora_config, MixLoraConfig):
        model_prefix_name = "mixlora"
        gate_layer_name = (
            f"mixlora.layers.{transformer_layer.layer_id_}.mlp.moe_gate.weight"
        )
        moe_layer_name_list = list(mlp_state_dict.keys())
    elif isinstance(lora_config, LoraMoeConfig):
        model_prefix_name = "loramoe"
        moe_layer_name_list = list(mlp_state_dict.keys())
    elif isinstance(lora_config, MolaConfig):
        model_prefix_name = "mola"
        moe_layer_name_list = list(all_state_dict.keys())
    else:
        model_prefix_name = "base_model.model.model"
        moe_layer_name_list = []

    # for fused MoEs such as MixLoRA
    mlp_moe_layer: LLMMoeBlock = transformer_layer.mlp_.moes_.get(
        lora_config.adapter_name, None
    )
    if mlp_moe_layer is not None:
        lora_weights[gate_layer_name] = mlp_moe_layer.gate_.weight

    for proj_name, lora_linear in all_state_dict.items():
        lora_linear: Linear
        if proj_name not in target_modules or not target_modules[proj_name]:
            continue
        module_name = (
            "self_attn"
            if proj_name in attn_state_dict
            else ("mlp" if proj_name in mlp_state_dict else None)
        )
        module_name = f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.{module_name}.{proj_name}"
        if proj_name in moe_layer_name_list:
            moe_layer = (
                lora_linear.moes_[lora_config.adapter_name]
                if lora_config.adapter_name in lora_linear.moes_
                else mlp_moe_layer
            )
            # for plugged MoEs such as LoRAMoE, MoLA, etc.
            if lora_config.adapter_name in lora_linear.moes_:
                lora_weights[f"{module_name}.moe_gate.weight"] = lora_linear.moes_[
                    lora_config.adapter_name
                ].gate_.weight

            for expert_idx in range(moe_layer.experts_):
                moe_lora_name = f"moe.{lora_config.adapter_name}.experts.{expert_idx}"
                lora_obj = lora_linear.loras_.get(moe_lora_name, None)
                if lora_obj is not None:
                    lora_weights[
                        f"{module_name}.experts.{expert_idx}.lora_A.weight"
                    ] = lora_obj.lora_a_.weight
                    lora_weights[
                        f"{module_name}.experts.{expert_idx}.lora_B.weight"
                    ] = lora_obj.lora_b_.weight

        else:
            lora_obj = lora_linear.loras_.get(lora_config.adapter_name, None)
            if lora_obj is not None:
                lora_weights[f"{module_name}.lora_A.weight"] = lora_obj.lora_a_.weight
                lora_weights[f"{module_name}.lora_B.weight"] = lora_obj.lora_b_.weight


def get_atten_tar(x, y, device, dtype):
    si = torch.arange(0, y, device=device, dtype = dtype)
    xi = torch.arange(1, x, device=device, dtype = dtype)#1~19
    lamb = torch.tensor(-2, device=device, dtype= dtype)
    alpha = (1 - torch.exp(-(si / 200))).detach()
    base = torch.empty(x-1, device=device, dtype= dtype)#(19)
    #base[0] = torch.log(torch.tensor(x, device=device, dtype = dtype)-1)
    base[0] = torch.exp(lamb)
    for i in range(1, x-1):
        #base[i] = base[i - 1] + torch.log(torch.tensor(x-i-1, device=device, dtype = dtype))
        base[i] = base[i - 1] + torch.exp(lamb * (i + 1))
    award = (0.1 * (0.5 - 1 / (xi + 1)) + 0.2).detach()
    #beta = (torch.log(x - xi) * award).expand(xi.shape[0], x-1).T
    beta = (torch.exp(lamb * xi) * award).expand(xi.shape[0], x-1).T
    beta = (beta / base).detach()

    

    return alpha, beta # alpha是从0开始的，beta[0]是1。至少321长度时，beta至少得0.8


class LLMModel(torch.nn.Module):
    def __init__(self, model: LLMForCausalLM):
        super().__init__()
        args: LLMModelConfig = model.config_
        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision."
            )
        self.model_ = model
        self.config_ = args
        # configs
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.device_ = args.device_
        self.dtype_ = args.dtype_

        self.attention_weight = torch.nn.Parameter(torch.empty(
            model.layers_[0].self_attn_.n_heads_,1,dtype=args.dtype_,device=args.device_,))
        
        self.routerup = torch.nn.Parameter(torch.empty(
            model.config_.dim_, 2,dtype=args.dtype_,device=args.device_,))
        """self. routerdown = torch.nn.Parameter(torch.empty(
            model.config_.dim_ * 2, 2,dtype=args.dtype_,device=args.device_,))"""
        self.cite_output = torch.nn.Parameter(torch.empty(
            model.config_.dim_,model.config_.dim_,dtype=args.dtype_,device=args.device_,))
        self.doc_proj = torch.nn.Parameter(torch.empty(
             model.config_.dim_, model.config_.dim_,dtype=args.dtype_,device=args.device_,))
        
        self.alpha, self.beta= get_atten_tar(40, 3000, args.device_, args.dtype_)
        self.silu = torch.nn.SiLU()

        self.output_ = OutputLayer()
        # adapter configs
        self.adapter_configs_: Dict[str, LoraConfig] = {}
    
    def token2id(self, t):
        if isinstance(t, torch.Tensor):
            x = t.item()
        else:
            x = t
        if x == 128002:
            return 0
        elif x == 128003:
            return 1
        elif x == 128004:
            return 2
        elif x == 128005:
            return 3
        elif x == 128008:
            return 4
        elif x >= 128010 and x <= 128255:
            return x - 128005
        else:
            return -1

    def attention_target(self, i, j, T):
        return self.alpha[j] * self.beta[T, i] * self.award[i]

    def _prepare_inputs(
        self, input_args: LLMModelInput, past_key_values: Optional[LLMCache] = None
    ):
        assert input_args.batch_tokens_ is not None, "Model have no input."
        assert (
            input_args.gradient_checkpoint_ == "none" or past_key_values is None
        ), "Cache is incompatible with gradient checkpointing."
        assert (
            not input_args.inference_mode_ or input_args.gradient_checkpoint_ == "none"
        ), "Can not use gradient checkpoint when inference."

        # prepare inputs
        if isinstance(input_args.batch_tokens_, torch.Tensor):
            input_ids = input_args.batch_tokens_.to(
                dtype=torch.int64, device=self.device_, requires_grad=False
            )
        else:
            input_ids = torch.tensor(
                input_args.batch_tokens_, dtype=torch.int64, device=self.device_, requires_grad=False
            )

        inputs_embeds = self.model_.embed_tokens(input_ids)

        """if input_ids.shape[-1] > 1:
            self.doc_embeds = []
            cites = input_args.batch_cites
            docs = input_args.batch_docs
            for doc in docs:
                doc = doc.clone().to(self.device_)
                doc = doc @ self.doc_proj
                self.doc_embeds.append(doc)
            for i, cite in enumerate(cites):
                for c in range(len(input_args.batch_cites_value[i])):
                    inputs_embeds[i, cite[c]] = self.doc_embeds[i][self.token2id(input_args.batch_cites_value[i][c]) - 1].to(self.device_)
        else:
            fk = self.token2id(input_ids[0,0])
            if fk != -1:
                inputs_embeds[0][0] = self.doc_embeds[0][fk - 1].to(self.device_)"""
        
        docs = input_args.batch_docs
        if input_ids.shape[-1] > 1:
            self.doc_embeds = []
            cites = input_args.batch_cites
            if not isinstance(docs[0][0], torch.Tensor):
                for i in range(len(docs)):
                    d = []
                    for j in range(len(docs[i])):
                        temp = self.model_.embed_tokens(torch.tensor(
                            docs[i][j][1:], dtype=torch.int64, device=self.device_, requires_grad=False))
                        temp = torch.mean(temp, dim = 0)
                        d.append(temp)
                    d = torch.stack(d)
                    self.doc_embeds.append(d)
            for i, cite in enumerate(cites):
                for c in range(len(input_args.batch_cites_value[i])):
                    doc_ind = self.token2id(input_args.batch_cites_value[i][c]) - 1
                    assert doc_ind >= 0, print("fake cite token")
                    inputs_embeds[i, cite[c]] = self.doc_embeds[i][doc_ind].to(self.device_)
        else:
            fk = self.token2id(input_ids[0,0]) - 1
            if fk >= 0:
                inputs_embeds[0][0] = self.doc_embeds[0][fk].to(self.device_) 
        
        if input_args.gradient_checkpoint_ != "none":
            inputs_embeds.requires_grad_(True)

        # prepare cache
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if past_seen_tokens is None:
            past_seen_tokens = 0

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        # prepare mask
        if input_args.batch_masks_ is not None:
            # 2d mask is passed through the layers
            if isinstance(input_args.batch_masks_, torch.Tensor):
                attention_mask = input_args.batch_masks_.to(
                    dtype=torch.int64, device=self.device_
                )
            else:
                attention_mask = torch.tensor(
                    input_args.batch_masks_, dtype=torch.int64, device=self.device_
                )
        else:
            attention_mask = None

        if self.config_.attn_implementation_ != "flash_attn":
            causal_mask = self.model_.causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values
            )
        else:
            causal_mask = attention_mask

        return input_ids, inputs_embeds, attention_mask, causal_mask, cache_position

    def _call_decoder_stack_original(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        # decoder layers
        num_adapters = len(input_args.batch_configs_)
        all_router_logits = [[] for _ in range(num_adapters)]
        gradient_checkpoint = CHECKPOINT_CLASSES[input_args.gradient_checkpoint_]

        for decoder_layer in self.model_.decoder_stack():
            hidden_states, *router_logits = gradient_checkpoint(
                decoder_layer.forward,
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                cache_position,
                past_key_value,
            )
            if len(router_logits) == 0:
                continue
            # collecting router logits
            assert len(router_logits) == num_adapters
            for idx in range(num_adapters):
                if router_logits[idx] is not None:
                    all_router_logits[idx].append(router_logits[idx])

        hidden_states = self.model_.norm(hidden_states)

        return hidden_states, all_router_logits


    def _call_decoder_stack(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
        #require_attention: Optional[int] = -1,
        #require_hide: Optional[int] = -1,
    ):
        # decoder layers
        gradient_checkpoint = CHECKPOINT_CLASSES[input_args.gradient_checkpoint_]

        #hidden_output = []
        #hidden_atten = []
        attention_matrixs = []
        for idx, decoder_layer in enumerate(self.model_.decoder_stack()):
            hidden_states, attention_matrix = gradient_checkpoint(
                decoder_layer.forward,
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                cache_position,
                past_key_value,
            )
            if idx in [31,30,29]:
                attention_matrixs.append(attention_matrix)
            """if require_hide == len(self.model_.layers_) or require_hide == idx:
                hidden_output.append(hidden_states)
            if require_attention == len(self.model_.layers_) or require_attention == idx:
                hidden_atten.append(hidden_attention)"""

        hidden_states = self.model_.norm(hidden_states)

        return hidden_states, attention_matrixs#hidden_atten, hidden_output

    # compute the model: output probs
    def forward(
        self, input_args: LLMModelInput, past_key_values: Optional[LLMCache] = None
    ) -> List[LLMModelOutput]:
        input_ids, inputs_embeds, attention_mask, causal_mask, cache_position = (
            self._prepare_inputs(input_args, past_key_values)
        )

        labels = input_args.batch_labels_

        input_args.batch_labels_ = None
        input_args.batch_tokens_ = None
        input_args.batch_masks_ = None

        # embed positions
        hidden_states = inputs_embeds

        rotary_emb = self.model_.rotary_embed(
            hidden_states, cache_position.unsqueeze(0)
        )

        hidden_states, attention_matrixs = self._call_decoder_stack(
            hidden_states,
            input_args,
            rotary_emb,
            causal_mask,
            cache_position,
            past_key_values,
            #require_attention,
            #require_hide,
        )
        attention_matrixs[-1] = attention_matrixs[-1].permute(0,2,3,1)
        attention_matrixs[-1] = torch.sum(attention_matrixs[-1], dim = -1).squeeze().to('cpu').detach()
        #print(attention_matrixs[-1].shape)
        #print(torch.mean(attention_matrixs[-1][input_args.batch_cites[0][0] + 1:input_args.batch_cites[0][2],input_args.batch_cites[0][0]]))
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        print(f"len:{input_args.batch_prompt_len[0]}")
        print(attention_matrixs[-1].shape)
        sns.heatmap(attention_matrixs[-1][input_args.batch_prompt_len[0]:,input_args.batch_prompt_len[0]:], annot=False, cmap="YlGnBu", vmin = 0, vmax = 0.2, xticklabels=False, yticklabels=False)
        plt.savefig("/yy21/heatmap", bbox_inches='tight', dpi=300)
        input()
        #route_logits = hidden_states @ (self.routerup @ self.routerdown)
        route_logits = hidden_states @ self.routerup
        hidden_cites = hidden_states @ self.cite_output
        norm_cite_logits = F.normalize(hidden_cites, p = 2, dim = 2)
        cite_logits = []
        for batch in range(hidden_states.shape[0]):
            #norm_doc = F.normalize(self.doc_embeds[batch], p = 2, dim = 1)
            norm_doc = F.normalize(self.doc_embeds[batch].detach(), p = 2, dim = 1)
            cite_logits.append(norm_cite_logits[batch] @ norm_doc.T)
            #cite_logits.append(norm_cite_logits[batch])
        

        # calculate loss
        output = self.output_(hidden_states, input_args)
        #att_s = hidden_atten[0].sum(dim = 1).squeeze() / 32 ###这里把List变为一个值
        assert isinstance(output, List)
        for indx, lora_config in enumerate(input_args.batch_configs_):
            output_data = output[indx]
            assert isinstance(output_data, LLMModelOutput)
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            output_data.batch_start_idx_ = start_idx
            output_data.batch_end_idx_ = end_idx
            #print(f"router:{route_logits[0,-1]}")
            #print(f"cite:{cite_logits}")
            if (labels is None) and (route_logits[0, -1, 1] > route_logits[0, -1, 0]):
                output_data.logits = cite_logits[0].unsqueeze(0)
                #output_data.logits = hidden_states[0].unsqueeze(0)
                output_data.cite_flag = True
            else:
                output_data.cite_flag = False
            if labels is None:
                continue
            # compute loss when labels provided
            output_data.loss = output_data.loss_fn_(
                input_ids[start_idx:end_idx],
                output_data.logits,
                labels[start_idx:end_idx],
                input_args.batch_cites,
                input_args.batch_cites_value,
                input_args.batch_prompt_len
            )
            output_data.loss_fn_ = None
            # route_logits和下面的合并
            for idx in range(len(input_args.batch_cites)):
                new_cites = []
                new_cites_v = []
                for i in range(len(input_args.batch_cites[idx])):
                    if input_args.batch_cites[idx][i] >= input_args.batch_prompt_len[idx]:
                        new_cites.append(input_args.batch_cites[idx][i])
                        if i < len(input_args.batch_cites_value[idx]):
                            new_cites_v.append(input_args.batch_cites_value[idx][i])
                input_args.batch_cites[idx] = new_cites
                input_args.batch_cites_value[idx] = new_cites_v
            if output_data.aux_loss is None:
                output_data.aux_loss = self.attn_mat_coin * self.attention_loss_fn(attention_matrixs, causal_mask, input_args.batch_cites, input_args.batch_prompt_len)
            else:
                output_data.aux_loss += self.attn_mat_coin * self.attention_loss_fn(attention_matrixs, causal_mask, input_args.batch_cites, input_args.batch_prompt_len)
            print(f"1:{output_data.aux_loss}")
            for idx in range(len(input_args.batch_cites)):
                if len(input_args.batch_cites[idx]) > len(input_args.batch_cites_value[idx]):
                    input_args.batch_cites[idx] = input_args.batch_cites[idx][:-1]
            output_data.aux_loss += self.router_coin * self.compute_route_loss(route_logits, input_args.batch_cites)#router的label中，cite位置的是1，其他是0
            print(f"2:{output_data.aux_loss}")
            #output_data.aux_loss += self.cite_coin * self.compute_cite_loss2(hidden_states, input_args.batch_cites,input_args.batch_cites_value,batch_doc_embed)#router的label中，cite位置的是1，其他是0
            output_data.aux_loss += self.cite_coin * self.compute_cite_loss(cite_logits, input_args.batch_cites,input_args.batch_cites_value)#router的label中，cite位置的是1，其他是0
            print(f"3:{output_data.aux_loss}")
        return output

    def from_pretrained(
        name_or_path: str,
        device: str,
        bits: int = None,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        load_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> "LLMModel":
        # load_dtype will change the precision of LLaMA pre-trained model
        # when loading with quantization (bits = 8 or bits = 4), load_dtype will only influence the actual computing precision
        if load_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported load dtype {load_dtype}")

        if compute_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported compute dtype {compute_dtype}")

        if load_dtype in [torch.bfloat16, torch.float16]:
            logging.info("Loading model with half precision.")

        # BFloat16 is only supported after Ampere GPUs
        if not executor.is_bf16_supported():
            if load_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                load_dtype = torch.float16

            if bits in [4, 8] and compute_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                compute_dtype = torch.float16

        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
            llm_model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map=device,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type,
                ),
                torch_dtype=load_dtype,
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=load_dtype,
            )

        llm_model.requires_grad_(False)

        model = from_pretrained(
            llm_model,
            attn_impl=attn_impl,
            use_sliding_window=use_sliding_window,
            device=device,
        )

        logging.info(f"Use {attn_impl} as attention implementation.")

        return LLMModel(model)

    def init_adapter(
        self, config: AdapterConfig, weight: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.attn_mat_coin = config.atten_coin
        self.router_coin = config.router_coin
        self.cite_coin = config.cite_coin
        # Patch for MixLoRA
        if isinstance(config, MixLoraConfig) and config.act_fn_ is None:
            config.act_fn_ = self.config_.hidden_act_

        self.adapter_configs_[config.adapter_name] = config
        # init output layer
        if config.task_name in task_dict and isinstance(
            task_dict[config.task_name], SequenceClassificationTask
        ):
            output_layer = ClassificationOutputLayer(
                **task_dict[config.task_name].init_kwargs(),
                hidden_size=self.config_.dim_,
                pad_token_id=self.config_.pad_token_id_,
                device=self.device_,
                weight=weight,
            )
        else:
            output_layer = CasualOutputLayer(
                vocab_size=self.config_.vocab_size_, weight=self.model_.lm_head_
            )
        
        if weight is None:
            torch.nn.init.kaiming_normal_(self.attention_weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.routerup, mode='fan_in', nonlinearity='relu')
            #torch.nn.init.kaiming_normal_(self.routerdown, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.cite_output, mode='fan_in', nonlinearity='relu')
            torch.nn.init.orthogonal_(self.doc_proj)
        else:
            with torch.no_grad():
                self.attention_weight.copy_(weight.get(f"{config.adapter_name}.attention_mat_weight", None))
                self.routerup.copy_(weight.get(f"{config.adapter_name}.router_weight_up", None))
                #self.routerdown.copy_(weight.get(f"{config.adapter_name}.router_weight_down", None))
                self.cite_output.copy_(weight.get(f"{config.adapter_name}.cite_weight", None))
                self.doc_proj.copy_(weight.get(f"{config.adapter_name}.doc_weight", None))
        self.output_.layers_[config.adapter_name] = output_layer
        if type(config) is not AdapterConfig:
            # init transformer layers
            for transformer_layer in self.model_.layers_:
                init_lora_layer_weight(transformer_layer, self.config_, config, weight)
        else:
            assert weight is None, "can not load basic adapter with weight"

        return config.adapter_name

    def get_adapter_weight_dict(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        # return the lora weight and target_module's name
        lora_weight_dict = self.output_.layers_[adapter_name].state_dict()
        atten_name = f"{adapter_name}.attention_mat_weight"
        lora_weight_dict[atten_name] = self.attention_weight
        route_name = f"{adapter_name}.router_weight_up"
        lora_weight_dict[route_name] = self.routerup
        """route_name = f"{adapter_name}.router_weight_down"
        lora_weight_dict[route_name] = self.routerdown"""
        cite_name = f"{adapter_name}.cite_weight"
        lora_weight_dict[cite_name] = self.cite_output
        doc_name = f"{adapter_name}.doc_weight"
        lora_weight_dict[doc_name] = self.doc_proj
        lora_config = self.adapter_configs_[adapter_name]
        for transformer_layer in self.model_.layers_:
            get_lora_layer_weight(transformer_layer, lora_config, lora_weight_dict)

        return lora_weight_dict

    def unload_adapter(
        self, adapter_name: str
    ) -> Tuple[LoraConfig, Dict[str, torch.Tensor]]:
        assert adapter_name in self.adapter_configs_, "adapter not exist"
        lora_weight = self.get_adapter_weight_dict(adapter_name)
        lora_config = self.adapter_configs_.pop(adapter_name)
        self.output_.layers_.pop(adapter_name)
        for transformer_layer in self.model_.layers_:
            attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
            attn_state_dict: Dict[str, torch.Tensor]
            mlp_state_dict: Dict[str, torch.Tensor]
            lora_layer_list = list(attn_state_dict.values())
            lora_layer_list.extend(mlp_state_dict.values())

            for lora_layer in lora_layer_list:
                if adapter_name in lora_layer.loras_:
                    lora_layer.loras_.pop(adapter_name, None)
                elif adapter_name in transformer_layer.mlp_.moes_:
                    for expert_idx in range(
                        transformer_layer.mlp_.moes_[adapter_name].experts_
                    ):
                        moe_lora_name = f"moe.{adapter_name}.experts.{expert_idx}"
                        lora_layer.loras_.pop(moe_lora_name, None)

                    transformer_layer.mlp_.moes_.pop(adapter_name)
                elif adapter_name in lora_layer.moes_:
                    for expert_idx in range(lora_layer.moes_[adapter_name].experts_):
                        moe_lora_name = f"moe.{adapter_name}.experts.{expert_idx}"
                        lora_layer.loras_.pop(moe_lora_name, None)

                    lora_layer.moes_.pop(lora_config.adapter_name, None)

        return lora_config, lora_weight

    def load_adapter(self, name_or_path: str, adapter_name: Optional[str] = None):
        if adapter_name is None:
            adapter_name = name_or_path

        if not os.path.exists(name_or_path):
            name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")
        with open(
            name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
        ) as fp:
            lora_config = lora_config_factory(json.load(fp))
        lora_config.adapter_name = adapter_name
        lora_weight = torch.load(
            name_or_path + os.sep + "adapter_model.bin",
            map_location=self.device_,
            weights_only=False,
        )

        self.init_adapter(lora_config, lora_weight)
        return adapter_name
    
    def compute_route_loss(self, logits, cites):
        nrom_logits = logits / torch.norm(logits, dim = -1, keepdim=True)
        b, l, v = logits.shape
        """for c in cites:
            if c[-1] == l:
                del c[-1]"""
        label = []
        for k in range(b):
            label.append([1 if i in cites[k] else 0 for i in range(l)])
        
        if isinstance(label, torch.Tensor):
            label = (
                label.clone()
                .detach()
                .to(dtype=torch.long, device=logits.device)
            )
        else:
            label = torch.tensor(label, dtype=torch.long, device=logits.device)

        loss_fn = torch.nn.CrossEntropyLoss() 
        return loss_fn(
            nrom_logits[..., :-1, :].contiguous().view(-1, v),
            label[..., 1:].contiguous().view(-1),
        )

    def compute_cite_loss2(self, logits, cites, cites_v, docs_pos):
        b = len(logits)
        docs_pos = [torch.tensor(i) for i in docs_pos]
        doc_embeds = []
        norm_logits = [F.normalize(logits[batch], p = 2, dim = 1) for batch in range(logits.shape[0])]
        for i in range(b):
            doc_embeds.append(norm_logits[i][docs_pos[i]].transpose(0,1))
        b_logits = []

        for i in range(len(cites)):
            b_logits.append(norm_logits[i] @ doc_embeds[i])
        for k in range(len(cites_v)):
            cites_v[k] = [self.token2id(i) for i in cites_v[k]]
        
        labels = []
        for k in range(b):
            labels.append([-100 for _ in range(logits[k].shape[0])])
            for i, v in zip(cites[k], cites_v[k]):
                labels[k][i] = v - 1

        if isinstance(labels[0], torch.Tensor):
            for k in range(b):
                labels[k] = (
                    labels[k].clone()
                    .detach()
                    .to(dtype=torch.long, device=logits[0].device)
                )
        else:
            for k in range(b):
                labels[k] = torch.tensor(labels[k], dtype=torch.long, device=logits[0].device)
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        
        loss = 0
        for k in range(b):
            if len(cites[k]) != 0:
                loss += loss_fn(
                b_logits[k][..., :-1, :].contiguous().view(-1, b_logits[k].shape[-1]),
                labels[k][..., 1:].contiguous().view(-1),
        )
        return loss / b

    def compute_cite_loss(self, logits, cites, cites_v):
        b = len(logits)

        for k in range(len(cites_v)):
            """if len(cites[k]) > len(cites_v[k]):
                del cites[k][-1]"""
            cites_v[k] = [self.token2id(i) for i in cites_v[k]]
        
        labels = []
        for k in range(b):
            labels.append([-100 for _ in range(logits[k].shape[0])])
            for i, v in zip(cites[k], cites_v[k]):
                labels[k][i] = v - 1

        if isinstance(labels[0], torch.Tensor):
            for k in range(b):
                labels[k] = (
                    labels[k].clone()
                    .detach()
                    .to(dtype=torch.long, device=logits[0].device)
                )
        else:
            for k in range(b):
                labels[k] = torch.tensor(labels[k], dtype=torch.long, device=logits[0].device)
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        
        loss = 0
        for k in range(b):
            if len(cites[k]) != 0:
                loss += loss_fn(
                logits[k][..., :-1, :].contiguous().view(-1, logits[k].shape[-1]),
                labels[k][..., 1:].contiguous().view(-1),
        )
        return loss / b


    def attention_loss_fn(self, mat, mask, cites, prompt_len):# cites: T个元素，每个元素代表c_i所在列
        mat = torch.stack(mat, dim = 0)
        mat = mat.permute(1,0,3,4,2)
        #final_mat = torch.matmul(mat, self.attention_weight).squeeze(-1)
        final_mat = torch.mean(mat, dim = -1)
        final_mat += mask
        final_mat = F.softmax(final_mat, dim=-1)
        loss = torch.tensor(0.0, dtype = final_mat.dtype, device = final_mat.device)
        num_layer = final_mat.shape[1]
        for batch in range(final_mat.shape[0]):
            if len(cites[batch]) == 0:
                continue
            for k in range(len(cites[batch]) - 1):
                for i in range(k + 1):
                    if cites[batch][k] == cites[batch][k + 1] - 1:
                        continue
                    loss_now = (self.alpha[cites[batch][k]:cites[batch][k + 1] - 1] * self.beta[k - i, k]).expand(1, num_layer,-1) - final_mat[batch,:,cites[batch][k]:cites[batch][k + 1] - 1,cites[batch][i]]
                    loss += F.relu(loss_now).sum() / (cites[batch][k + 1] - cites[batch][k])

        return loss
