import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel
from huggingface_hub import login


def load_tokenizer(model_name):
    if "LLaMa-2-7B" in model_name or "LLaMa-2-13B" in model_name:
        return LlamaTokenizer.from_pretrained(model_name)
    elif "chatglm" in model_name or "Yi-6B" in model_name or "Ziya2-13B-Base" in model_name:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name, padding_side='left')
    # return AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    if "chatglm" in model_name or "Yi-6B" in model_name:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    # import pdb
    # pdb.set_trace()

    if "chatglm" in model_name:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **model_args).eval()
    elif "Yi-6B" in model_name or "Ziya2-13B-Base" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval()
    else:
        if "Llama-3-8b" in model_name or "Qwen1.5-7B" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device
