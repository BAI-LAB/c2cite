import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import re
import matplotlib.pyplot as plt

from moe_peft.common import LLMBatchConfig, LLMModelInput, Tokens, cache_factory
from moe_peft.executors import executor
from moe_peft.model import LLMModel
from moe_peft.prompter import Prompter
from moe_peft.tokenizer import Tokenizer
from moe_peft.solutions import get_output


@dataclass
class GenerateData:
    adapter_name_: str = None
    prompt_index_: int = None
    prefix_length_: int = None
    raw_tokens_: Tokens = None


@dataclass
class GenerateConfig:
    adapter_name: str = None
    prompts: List[Union[str, Tuple[str, str]]] = None
    prompt_template: str = None
    # Generate Arguments
    batch_size: int = 8
    stop_token: str = None
    temperature: float = 1
    top_p: float = 0.9
    top_k: float = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    renormalize_logits: bool = True
    # Do not set these manually
    prompter_: Prompter = None
    stop_token_: torch.Tensor = None
    data_: List[GenerateData] = None

    # Set prompt_template_ to enable the prompter
    def generate_prompt(self, instruction: str, input: str = None) -> str:
        if self.prompter_ is None:
            self.prompter_ = Prompter(self.prompt_template)

        return self.prompter_.generate_prompt(instruction=instruction, input=input)

    def get_prompts(self) -> List[str]:
        prompts = []
        for prompt in self.prompts:
            args = prompt if isinstance(prompt, Tuple) else (prompt, None)
            prompts.append(self.generate_prompt(*args))

        return prompts

    def get_response(self, output: str) -> str:
        if self.prompter_ is None:
            return output.strip()
        else:
            return self.prompter_.get_response(output)

    def reset_parameters(self):
        self.prompter_ = Prompter(self.prompt_template)
        self.stop_token_ = None
        self.data_ = []


def _logits_sample_top_p(probs, p, filter_value=float("-inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(probs, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return probs.masked_fill(indices_to_remove, filter_value)


def _logits_sample_top_k(probs, k, filter_value=float("-inf")):
    top_k = min(k, probs.size(-1))  # Safety check
    indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
    return probs.masked_fill(indices_to_remove, filter_value)


def _logits_repetition_penalty(prev_tokens, probs, penalty):
    score = torch.gather(probs, 1, prev_tokens)
    score = torch.where(score < 0, score * penalty, score / penalty)
    probs.scatter_(1, prev_tokens, score)
    return probs


def id2token(x):
    if x == 0:
        return 128002
    elif x == 1:
        return 128003
    elif x == 2:
        return 128004
    elif x == 3:
        return 128005
    elif x == 4:
        return 128008
    elif x >= 5:
        return 128005 + x
    else:
        assert False, "wrong router"

def logits_process(
    probs: torch.Tensor,
    prev_tokens: torch.Tensor,
    cite_flag = False,
    temperature=0.9,
    top_p=0,
    top_k=0,
    do_sample=True,
    repetition_penalty=1.01,
    renormalize_logits=True,
):
    if cite_flag == False:
        process_conditions = any([repetition_penalty > 0])
        sample_conditions = any([temperature > 0, top_p > 0 and top_p <= 1.0, top_k > 0])

        if not do_sample and sample_conditions:
            do_sample = True
            logging.warn("do_sample force to enabled.")

        if repetition_penalty > 0:
            probs = _logits_repetition_penalty(prev_tokens, probs, repetition_penalty)

        if process_conditions and renormalize_logits:
            probs = probs.log_softmax(-1)

        if temperature > 0:
            probs = probs / temperature

        if top_k > 0:
            probs = _logits_sample_top_k(probs, top_k)

        if top_p > 0 and top_p <= 1.0:
            probs = _logits_sample_top_p(probs, top_p)

        if sample_conditions and renormalize_logits:
            probs = probs.log_softmax(-1)
    else:
        do_sample = False

    if do_sample:
        probs = torch.softmax(probs, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(probs, dim=-1)
    
    if cite_flag:
        for i in range(probs.shape[0]):
            next_token[i] = id2token(next_token[i] + 1)
    return next_token.reshape(-1)


def _extract_effective_tokens(
    tokenizer: Tokenizer,
    prefix_length: int,
    tokens: Tokens,
    remove_prefix=True,
    remove_pad=True,
    remove_eos=True,
):
    if remove_prefix:
        tokens = tokens[prefix_length:]

    if remove_pad and tokenizer.pad_id_ in tokens:
        pad_idx = tokens.index(tokenizer.pad_id_)
        tokens = tokens[:pad_idx]

    if remove_eos and tokenizer.eos_id_ in tokens:
        stop_idx = tokens.index(tokenizer.eos_id_)
        tokens = tokens[:stop_idx]

    return tokens


def _gen_outputs(
    tokenizer: Tokenizer,
    config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    tokens: torch.Tensor,
):
    tokens = tokens.tolist()
    packed_outputs: Dict[str, List[str]] = {}
    for idx, data in enumerate(current_jobs):
        output = config_dict[data.adapter_name_].get_response(
            tokenizer.decode(
                _extract_effective_tokens(
                    tokenizer,
                    data.prefix_length_,
                    tokens[idx],
                    remove_prefix=True,
                    remove_pad=True,
                    remove_eos=True,
                )
            )
        )
        if data.adapter_name_ in packed_outputs:
            packed_outputs[data.adapter_name_].append(output)
        else:
            packed_outputs[data.adapter_name_] = [output]

    return packed_outputs


def _dispatch_task_in(
    configs: List[GenerateConfig],# config.data_, config.batch_size, config, config.adapter_name
    concurrent_jobs: int,
    strategy: str = "fair",
):
    assert strategy in ["fair", "fifo"], f"Unknown dispatch strategy {strategy}"
    current_jobs = []
    batch_config = []
    input_tokens = []
    max_tokens_len = 0
    min_tokens_len = sys.maxsize
    for config in configs:
        if len(batch_config) >= concurrent_jobs:
            break

        if len(config.data_) == 0:
            continue

        if strategy == "fair":
            per_task_jobs = max(concurrent_jobs // len(configs), 1)
        else:
            per_task_jobs = concurrent_jobs

        per_task_jobs = min(per_task_jobs, config.batch_size)

        batch_start_idx = len(input_tokens)
        while per_task_jobs > 0 and len(config.data_) > 0:
            per_task_jobs = per_task_jobs - 1
            data = config.data_.pop(0)
            current_jobs.append(data)
            tokens = data.raw_tokens_
            max_tokens_len = max(len(tokens), max_tokens_len)
            min_tokens_len = min(len(tokens), min_tokens_len)
            input_tokens.append(tokens)

        batch_config.append(
            LLMBatchConfig(
                adapter_name_=config.adapter_name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=len(input_tokens),
            )
        )

    return (
        current_jobs,
        batch_config,
        input_tokens,
        max_tokens_len,
        min_tokens_len,
    )


def _dispatch_task_out(
    tokenizer: Tokenizer,
    # config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    tokens: torch.Tensor,
    stop_reached: torch.Tensor,
    attentions,
    hides,
    require_attention,
    require_hide
):
    """hide = []
    if require_hide != -1:
        ans_len = len(hides)
        for i in range(len(hides[0])):
            hide.append(torch.cat([t[i] for t in hides], dim = 1))
    if require_attention != -1:
        ans_len = len(attentions)
        for i in range(len(hides[0])):
            hide.append(torch.cat([t[i] for t in attentions], dim = 1))"""
    tokens = tokens.tolist()
    stop_reached = stop_reached.view(-1).tolist()
    packed_outputs: List[str] = []
    packed_add = []
    running_jobs: List[GenerateData] = []
    for idx, data in enumerate(current_jobs): # 这里的data是evaluate data, 但是应该是generate data
        if stop_reached[idx]:
            output_tokens = _extract_effective_tokens(
                        tokenizer,
                        data.prefix_length_,
                        tokens[idx],
                        remove_prefix=True,
                        remove_pad=True,
                        remove_eos=True,
                    )
            #if len(hide):
            #    get_output(hide, output_tokens, ans_len)
            output_s = tokenizer.decode(output_tokens).strip()
            output = re.sub(r'<\|reserved_special_token_(\d+)\|>', r'[\1]', output_s)
            packed_outputs.append(output)
        else:
            data.tokens = _extract_effective_tokens(
                tokenizer,
                data.prefix_length_,
                tokens[idx],
                remove_prefix=False,
                remove_pad=True,
                remove_eos=False,
            )
            running_jobs.append(data)

    return packed_outputs, running_jobs


def _batch_generate(
    model: LLMModel,
    tokenizer: Tokenizer,
    max_gen_len: Optional[int],
    use_cache: bool,
    require_attention: Optional[int],
    require_hide: Optional[int],
    cache_implementation: Optional[str],
    stream_callback: Optional[Callable],
    #config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    batch_config: List[LLMBatchConfig],
    input_tokens: List[Tokens],
    max_tokens_len: int,
    min_tokens_len: int,
):
    executor.empty_cache()
    device = torch.device(model.device_)
    batch_size = len(input_tokens)
    if max_gen_len is None:
        max_gen_len = model.config_.max_seq_len_ - max_tokens_len
    total_len = min(model.config_.max_seq_len_, max_gen_len + max_tokens_len)
    past_key_values = (
        cache_factory(
            cache_implementation=cache_implementation,
            config=model.model_.model_config(),
            batch_size=batch_size,
            max_cache_len=total_len,
        )
        if cache_implementation is not None
        else None
    )

    tokens = torch.full(
        (batch_size, total_len), tokenizer.pad_id_, dtype=torch.int64, device=device
    )
    # print(f"yyyyyy:\n{tokenizer.decode(input_tokens[0])}")
    for k, t in enumerate(input_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)
    def condition(i):
        return (128010 <= i <= 128255) or i in {128005, 128004, 128003, 128002, 128008}
    prompt_len = len(input_tokens[0])
    cite = [index for index, value in enumerate(input_tokens[0]) if condition(value)]
    cite_v = [value for value in input_tokens[0] if condition(value)]

    prev_pos = 0
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_

    hidden_states = []
    hidden_attentions = []
    #arti_mask = torch.ones(batch_size, total_len, device=device, dtype=torch.int64)
    cite_start = -1
    #flag = -1
    plac = []
    for cur_pos in range(min_tokens_len, total_len):
        input_data = LLMModelInput(            
            batch_configs_=batch_config,
            batch_tokens_=tokens[:, prev_pos:cur_pos].tolist(),
            #batch_masks_ = arti_mask,############
            batch_cites = [cite],
            batch_cites_value = [cite_v],
            batch_docs = [current_jobs[0].citation_tokens],
            batch_prompt_len = [prompt_len],
            inference_mode_=True,
        )
        # print(f"fuck:\n{tokenizer.decode(tokens[0, prev_pos:cur_pos])}")
        outputs = model.forward(input_data, past_key_values)
        #hidden_states.append(hidden_state)
        #hidden_attentions.append(hidden_attention)
        
        #if flag != -1:
            #输出attention

        for output in outputs:
            #config = config_dict[output.adapter_name]
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_

            next_token = logits_process(
                output.logits[:, -1],#####看看它的维度,这里是乘完doc的，应该是logits
                tokens[start_idx:end_idx, :cur_pos],
                cite_flag = output.cite_flag,
            )

            next_token = torch.where(
                input_text_mask[start_idx:end_idx, cur_pos],
                tokens[start_idx:end_idx, cur_pos],
                next_token,
            ).to(torch.int64)
            #print(tokenizer.decode(next_token))
            if output.cite_flag == True:# 记得查看input_text_mask的形状
                for i in range(start_idx, end_idx):
                    if input_text_mask[i, cur_pos]:#纯废话，这时候考虑上多batch了
                        continue
                    cite.append(cur_pos)
                    cite_v.append(next_token)

            tokens[start_idx:end_idx, cur_pos] = next_token
            stop_criteria = (~input_text_mask[start_idx:end_idx, cur_pos]) & (
                next_token == torch.tensor(
                [tokenizer.eos_id_], dtype=torch.int64, device=device
                )
            )
            stop_reached[start_idx:end_idx] |= stop_criteria
            if cite_start != -1:
                if tokenizer.decode(next_token)[-1] in ['.','!','?']:
                    #arti_mask[start_idx:end_idx, cite_start:cur_pos] = 0
                    #tokens[start_idx:end_idx, cur_pos] = tokenizer.encode(tokenizer.decode(next_token)[-1])[-1]
                    cite_start = -1
                if tokenizer.decode(next_token)[-1] in ['0','1','2','3','4','5','6','7','8','9']:
                    plac.append(cur_pos)
                    # tokens[start_idx:end_idx, cur_pos] = (tokens[start_idx:end_idx, cur_pos] + 2)

            if tokenizer.decode(next_token)[-1] == '[' or tokenizer.decode(next_token) == '[':
                if cite_start == -1:
                    cite_start = cur_pos
                #flag = cur_pos
        
        stop_reached |= total_len - cur_pos == 1

        if any(stop_reached):
            break

        if use_cache:
            prev_pos = cur_pos

    """input_data = LLMModelInput(
            batch_configs_=batch_config,
            batch_tokens_=tokens[:,:hidden_attention.shape[0]].tolist(),
            inference_mode_=True,
    )"""
    # print(f"fuck:\n{tokenizer.decode(tokens[0, prev_pos:cur_pos])}")
    #outputs, _, attn = model.forward(input_data, None, require_attention, require_hide)
    """for i in plac:

        plt.figure(figsize=(hidden_attention.shape[0], 5), dpi = 50)
        print("painting")
        plt.bar(range(hidden_attention.shape[0]), attn[:,i].cpu().numpy())
        plt.xticks(range(hidden_attention.shape[0]), [tokenizer.decode(j) for j in tokens[0][:hidden_attention.shape[0]]], fontsize = 8)
        plt.savefig("high_res_heatmap.svg", dpi=50)
        print("ok~")
        input()
    """
    """attn[torch.arange(hidden_attention.shape[0]), torch.arange(hidden_attention.shape[0])] = 0.0
    attn = torch.nn.functional.normalize(attn, p=2, dim=1)
    attn = attn[min_tokens_len:hidden_attention.shape[0],min_tokens_len:hidden_attention.shape[0]]

    plt.figure(figsize=(hidden_attention.shape[0] - min_tokens_len, hidden_attention.shape[0] - min_tokens_len))  # 调整图像大小
    plt.imshow(attn.cpu().numpy(), cmap='viridis', vmin = 0, vmax = 0.1)
    plt.colorbar(label='Value')
    plt.xticks(range(hidden_attention.shape[0] - min_tokens_len), [tokenizer.decode(i) for i in tokens[0][min_tokens_len:hidden_attention.shape[0]]], fontsize = 10)
    plt.yticks(range(hidden_attention.shape[0] - min_tokens_len), [tokenizer.decode(i) for i in tokens[0][min_tokens_len:hidden_attention.shape[0]]], fontsize = 10)
    plt.savefig("high_res_heatmap.png", dpi=200)  # 保存为高分辨率图像
    plt.show()
    print("ok~")
    input()"""
    """token2 = tokens * arti_mask
    lst = token2[0].tolist()
    lst = [ele for ele in lst if ele != 0]
    tokens = torch.tensor(lst, dtype=torch.int64, device=device).unsqueeze(0)"""

    return _dispatch_task_out(
        tokenizer, current_jobs, tokens, stop_reached, hidden_states, hidden_attentions, require_attention, require_hide
    )


def _batch_generate_original(
    model: LLMModel,
    tokenizer: Tokenizer,
    max_gen_len: Optional[int],
    use_cache: bool,
    cache_implementation: Optional[str],
    stream_callback: Optional[Callable],
    config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    batch_config: List[LLMBatchConfig],
    input_tokens: List[Tokens],
    max_tokens_len: int,
    min_tokens_len: int,
):
    executor.empty_cache()
    device = torch.device(model.device_)
    batch_size = len(input_tokens)
    if max_gen_len is None:
        max_gen_len = model.config_.max_seq_len_ - max_tokens_len
    total_len = min(model.config_.max_seq_len_, max_gen_len + max_tokens_len)

    past_key_values = (
        cache_factory(
            cache_implementation=cache_implementation,
            config=model.model_.model_config(),
            batch_size=batch_size,
            max_cache_len=total_len,
        )
        if cache_implementation is not None
        else None
    )

    tokens = torch.full(
        (batch_size, total_len), tokenizer.pad_id_, dtype=torch.int64, device=device
    )
    for k, t in enumerate(input_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)

    prev_pos = 0
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_
    for cur_pos in range(min_tokens_len, total_len):
        input_data = LLMModelInput(
            batch_configs_=batch_config,
            batch_tokens_=tokens[:, prev_pos:cur_pos].tolist(),
            inference_mode_=True,
        )
        outputs = model.forward(input_data, past_key_values)
        for output in outputs:
            config = config_dict[output.adapter_name]
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_

            next_token = logits_process(
                output.logits[:, -1],
                tokens[start_idx:end_idx, :cur_pos],
                config.temperature,
                config.top_p,
                config.top_k,
                config.do_sample,
                config.repetition_penalty,
                config.renormalize_logits,
            )

            next_token = torch.where(
                input_text_mask[start_idx:end_idx, cur_pos],
                tokens[start_idx:end_idx, cur_pos],
                next_token,
            ).to(torch.int64)
            tokens[start_idx:end_idx, cur_pos] = next_token
            stop_criteria = (~input_text_mask[start_idx:end_idx, cur_pos]) & (
                next_token == config.stop_token_
            )
            stop_reached[start_idx:end_idx] |= stop_criteria

        stop_reached |= total_len - cur_pos == 1

        if any(stop_reached):
            break

        if stream_callback is not None:
            stream_callback(
                cur_pos,
                _gen_outputs(
                    tokenizer,
                    config_dict,
                    current_jobs,
                    tokens,
                ),
            )

        if use_cache:
            prev_pos = cur_pos

    return _dispatch_task_out(
        tokenizer, config_dict, current_jobs, tokens, stop_reached
    )


@torch.inference_mode()
def generate(
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[GenerateConfig],
    max_gen_len: Optional[int] = None,
    use_cache: bool = True,
    dispatch_strategy: str = "fair",
    concurrent_jobs: Optional[int] = None,
    cache_implementation: Optional[str] = None,
    stream_callback: Optional[Callable] = None,
):
    if concurrent_jobs is None:
        concurrent_jobs = len(configs)
        logging.info(f"Setting concurrent jobs to {concurrent_jobs} automatically")

    assert concurrent_jobs > 0

    # prepare for generation
    device = torch.device(model.device_)
    config_dict = {}
    for config in configs:
        config.reset_parameters()
        config_dict[config.adapter_name] = config
        if config.stop_token is not None:
            stop_token = tokenizer.encode(" " + config.stop_token, False)[-1]
        else:
            stop_token = tokenizer.eos_id_
        config.stop_token_ = torch.tensor(
            [stop_token], dtype=torch.int64, device=device
        )
        for idx, prompt in enumerate(config.prompts):
            args = prompt if isinstance(prompt, Tuple) else (prompt, None)
            tokens = tokenizer.encode(config.generate_prompt(*args))
            assert (
                len(tokens) < model.config_.max_seq_len_
            ), "Inputs exceeded max sequence length of model."
            config.data_.append(
                GenerateData(
                    adapter_name_=config.adapter_name,
                    prompt_index_=idx,
                    prefix_length_=len(tokens),
                    raw_tokens_=tokens,
                )
            )

    if use_cache and cache_implementation is None:
        cache_implementation = model.model_.cache_implementation()
        if cache_implementation is None:
            logging.warn(
                "Cache disabled by model, use cache_implementation to force enable."
            )
            use_cache = False

    packed_outputs: Dict[str, List] = {}

    while True:# configs里的data在变，是调度的唯一指标
        dispatch_args = _dispatch_task_in(configs, concurrent_jobs, dispatch_strategy)
        # 包含：current_jobs, batch_config(LLMBatchConfig(taskname,start,end)),
        # batch_tokens, max_lenth, min_length
        if len(dispatch_args[0]) == 0:
            break

        outputs, running_jobs = _batch_generate(
            model,
            tokenizer,
            max_gen_len,
            use_cache,
            cache_implementation,
            stream_callback,
            config_dict,
            *dispatch_args,
        )

        for name, output in outputs.items():
            if name in packed_outputs:
                packed_outputs[name].extend(output)
            else:
                packed_outputs[name] = output

        for data in running_jobs:
            config_dict[data.adapter_name_].data_.append(data)

    return packed_outputs
