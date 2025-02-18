import base64
import copy
import json
import os
import struct
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union
import aioboto3
import aiohttp
import numpy as np
import ollama
import torch
import pickle
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    api_key='sk-'
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if model.startswith("q")or model.startswith("l"):
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        os.environ["OPENAI_API_KEY"] = "sk-"
    else: 
        base_url='https://api.chatanywhere.com.cn/v1'
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})


    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")
    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
    os.environ["OPENAI_API_KEY"] = api_key
    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        content = response.choices[0].message.content
        if content is None:
            return ""
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    api_version=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    return content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )
    kwargs.pop("hashing_kv", None)
    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

    return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response_text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False
    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("hashing_kv", None)
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    if stream:
        """cannot cache stream response"""

        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        return response["message"]["content"]


@lru_cache(maxsize=1)
def initialize_lmdeploy_pipeline(
    model,
    tp=1,
    chat_template=None,
    log_level="WARNING",
    model_format="hf",
    quant_policy=0,
):
    from lmdeploy import pipeline, ChatTemplateConfig, TurbomindEngineConfig

    lmdeploy_pipe = pipeline(
        model_path=model,
        backend_config=TurbomindEngineConfig(
            tp=tp, model_format=model_format, quant_policy=quant_policy
        ),
        chat_template_config=(
            ChatTemplateConfig(model_name=chat_template) if chat_template else None
        ),
        log_level="WARNING",
    )
    return lmdeploy_pipe


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def lmdeploy_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    chat_template=None,
    model_format="hf",
    quant_policy=0,
    **kwargs,
) -> str:
    """
    Args:
        model (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        chat_template (str): needed when model is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on,
            and when the model name of local path did not match the original model name in HF.
        tp (int): tensor parallel
        prompt (Union[str, List[str]]): input texts to be completed.
        do_preprocess (bool): whether pre-process the messages. Default to
            True, which means chat_template will be applied.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        do_sample (bool): Whether or not to use sampling, use greedy decoding otherwise.
            Default to be False, which means greedy decoding will be applied.
    """
    try:
        import lmdeploy
        from lmdeploy import version_info, GenerationConfig
    except Exception:
        raise ImportError("Please install lmdeploy before initialize lmdeploy backend.")
    kwargs.pop("hashing_kv", None)
    kwargs.pop("response_format", None)
    max_new_tokens = kwargs.pop("max_tokens", 512)
    tp = kwargs.pop("tp", 1)
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    do_preprocess = kwargs.pop("do_preprocess", True)
    do_sample = kwargs.pop("do_sample", False)
    gen_params = kwargs

    version = version_info
    if do_sample is not None and version < (0, 6, 0):
        raise RuntimeError(
            "`do_sample` parameter is not supported by lmdeploy until "
            f"v0.6.0, but currently using lmdeloy {lmdeploy.__version__}"
        )
    else:
        do_sample = True
        gen_params.update(do_sample=do_sample)

    lmdeploy_pipe = initialize_lmdeploy_pipeline(
        model=model,
        tp=tp,
        chat_template=chat_template,
        model_format=model_format,
        quant_policy=quant_policy,
        log_level="WARNING",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    gen_config = GenerationConfig(
        skip_special_tokens=skip_special_tokens,
        max_new_tokens=max_new_tokens,
        **gen_params,
    )

    response = ""
    async for res in lmdeploy_pipe.generate(
        messages,
        gen_config=gen_config,
        do_preprocess=do_preprocess,
        stream_response=False,
        session_id=1,
    ):
        response += res.response
    return response


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


async def openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "qwen2.5-3b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat

    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

import torch.functional as F

def scoring(all_label_repre,sen_emb,label_id = 0,label_num = 4):
    sim = []
    for lr in all_label_repre:
        csim = F.cosine_similarity(lr.unsqueeze(1), sen_emb.unsqueeze(0), dim=2)
        max_values, _ = torch.max(csim, dim=1)

        kvs,kinds = torch.topk(max_values,k = len(max_values))

        ss = torch.mean(kvs)
        sim.append(float(ss))
    sim = torch.tensor(sim)


    s = sim[label_id] * (label_num - 1)/(torch.sum(sim) - sim[label_id])
    # print('final sim:', sim,'final score:',s)
    return s

model=None
tokenizer=None
control_center=None

def de_noising(class_repre,vec,label_id = 0,label_num = 4):
    sim = F.cosine_similarity(class_repre, vec.unsqueeze(0), dim=1).squeeze()
    # print('sim:',sim)

    s = sim[label_id] * (label_num - 1)/(torch.sum(sim) - sim[label_id])
    return s
async def llama_8b_31_complete(
    prompt, system_prompt="", history_messages=[], keyword_extraction=False, controls=None, **kwargs
) -> str:
    global model, tokenizer, control_center
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_dpo = False
    if model is None:
        if use_dpo:
            model = LlamaForCausalLM.from_pretrained("path2dpo").to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained("path2llama").to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("path2llama")
    if control_center is None:
        control_center = pickle.load(open("path2reformatted_data.pkl", "rb"))

    class_ids = tokenizer.convert_tokens_to_ids(controls[0])
    emb_weight=model.model.embed_tokens.weight
    class_repre = emb_weight[class_ids]
    each_prompt_gen = []
    trial_number = 0
    current_input = prompt
    length = len(tokenizer.encode(current_input))
    max_length = length + 512
    old_weight = 0.0
    def set_value_activations(values_per_layer, coef_value=3):
        def value_activation_replacement_hook(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = coef_val

            return hook

        hooks = []

        for layer in range(model.config.n_layer):
            if layer in values_per_layer:
                values = values_per_layer[layer]
            else:
                values = []

            hook = model.transformer.h[layer].mlp.c_fc.register_forward_hook(
                value_activation_replacement_hook(values, coef_value)
            )

            hooks.append(hook)

        return hooks
    ids = []
    label_ids = {}
    all_label_repre = []
    all_control = {}
    for w in controls:
        cw = w.strip()
        cw_id = [tokenizer.convert_tokens_to_ids(cw)]
        if cw_id[0] == None:
            cw=cw[1:]
            cw_id = [tokenizer.convert_tokens_to_ids(cw)]
            if cw_id[0] == None:
                print('Error:',cw)
                continue
        filter_socre = de_noising(class_repre,emb_weight[cw_id[0]],label_count,4)
        # print('filter_socre:',filter_socre,'token:',tokenizer.decode(cw_id[0]))
        if len(cw_id) < 2 and cw_id[0] != 50267 and filter_socre > 1.0:
            # print('cw_id:', cw_id, tokenizer.decode(cw_id))
            ids.append(cw_id[0])
    label_ids[label_count] = ids
    label_repre = emb_weight[ids]
    all_label_repre.append(label_repre)
    label_count = label_count + 1
    for control in controls:
        if control not in control_center:
            print('miss control',control)
            continue
        control_layers = control_center[control]
        print('add control',control) 
        for layer in control_layers:
            if layer[0][0] in all_control:
                all_control[layer[0][0]].append(layer[0][1])
            else:
                all_control[layer[0][0]]=[layer[0][1]]
    while len(each_prompt_gen) < 1:
        trial_number += 1
        all_weight = []


        while length < max_length:
            ot_ids = tokenizer.encode(current_input)
            ot_repre = emb_weight[ot_ids]  
            sen_score_topic = scoring(all_label_repre, ot_repre)  # 假设 scoring 函数已定义
            token_score_topic = scoring(all_label_repre, ot_repre[-1, :].unsqueeze(0))
            final_score = torch.max(sen_score_topic, token_score_topic)

            if 1.10- final_score > 0:
                new_weight = 0.3 / (1.0 + torch.exp(-torch.tensor(1.10 - final_score) * (length)))
            else:
                new_weight = 1e-20

            new_weight = (new_weight + old_weight) / 2.0
            old_weight = new_weight

            all_weight.append(float(new_weight))

            # 移除旧的钩子
            for hook in hooks:
                hook.remove()

            # 为每一层添加新的钩子，并使用动态权重
            hooks = set_value_activations(all_control, new_weight)

            with torch.no_grad():
                outputs = model.generate([current_input], max_new_tokens=1, num_return_sequences=1, do_sample=True)
            current_input += tokenizer.decode(outputs[0], skip_special_tokens=True)
            length = len(tokenizer.encode(current_input))

        split_ = current_input.split('\n')
        if len(split_) > 2:
            continue

        each_prompt_gen.append(current_input)

    # 移除所有钩子
    for hook in hooks:
        hook.remove()

    return each_prompt_gen[0]


async def nvidia_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        "conversation-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await bedrock_complete_if_cache(
        "anthropic.claude-3-haiku-20240307-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    result = await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    base_url='https://api.chatanywhere.com.cn/v1'
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def fetch_data(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()
            data_list = response_json.get("data", [])
            return data_list


async def jina_embedding(
    texts: list[str],
    dimensions: int = 1024,
    late_chunking: bool = False,
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["JINA_API_KEY"] = api_key
    url = "https://api.jina.ai/v1/embeddings" if not base_url else base_url
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
    }
    data = {
        "model": "jina-embeddings-v3",
        "normalized": True,
        "embedding_type": "float",
        "dimensions": f"{dimensions}",
        "late_chunking": late_chunking,
        "input": texts,
    }
    data_list = await fetch_data(url, headers, data)
    return np.array([dp["embedding"] for dp in data_list])


@wrap_embedding_func_with_attrs(embedding_dim=2048, max_token_size=512)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def nvidia_openai_embedding(
    texts: list[str],
    model: str = "nvidia/llama-3.2-nv-embedqa-1b-v1",
    # refer to https://build.nvidia.com/nim?filters=usecase%3Ausecase_text_to_embedding
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: str = None,
    input_type: str = "passage",  # query for retrieval, passage for embedding
    trunc: str = "NONE",  # NONE or START or END
    encode: str = "float",  # float or base64
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model,
        input=texts,
        encoding_format=encode,
        extra_body={"input_type": input_type, "truncate": trunc},
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    api_version: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "netease-youdao/bce-embedding-base_v1",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 512,
    api_key: str = None,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)


# @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),  # TODO: fix exceptions
# )
async def bedrock_embedding(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
) -> np.ndarray:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
                            # 'dimensions': embedding_dim,
                            "embeddingTypes": ["float"],
                        }
                    )
                elif "v1" in model:
                    body = json.dumps({"inputText": text})
                else:
                    raise ValueError(f"Model {model} is not supported!")

                response = await bedrock_async_client.invoke_model(
                    modelId=model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = await response.get("body").json()

                embed_texts.append(response_body["embedding"])
        elif model_provider == "cohere":
            body = json.dumps(
                {"texts": texts, "input_type": "search_document", "truncate": "NONE"}
            )

            response = await bedrock_async_client.invoke_model(
                model=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            embed_texts = response_body["embeddings"]
        else:
            raise ValueError(f"Model provider '{model_provider}' is not supported!")

        return np.array(embed_texts)


async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()



class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


