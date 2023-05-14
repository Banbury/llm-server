import os

import inspect
import accelerate
import torch
import transformers
from safetensors.torch import load_file as safe_load
from quant import make_quant
from modelutils import find_layers

import torch
import torch.nn as nn

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig

from yaspin import yaspin

# model_path = "d:\\git\\oobabooga_windows\\text-generation-webui\\models\\wizard-vicuna-13B-GPTQ-4bit"
# checkpoint = "wizard-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"
# model_path = "d:\\git\\oobabooga_windows\\text-generation-webui\\models\\WizardLM-13B-Uncensored"
# checkpoint = "4bit-128g.safetensors"
DEV = torch.device('cuda:0')

@yaspin(text="Loading tokenizer...")
def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)

@yaspin(text="Loading model...")
def load_quantized(model_path, checkpoint):
    def noop(*args, **kwargs):
        pass

    checkpoint_path = os.path.join(model_path, checkpoint)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    torch.set_default_dtype(torch.float)
    model = model.eval()
    gptq_args = inspect.getfullargspec(make_quant).args

    layers = find_layers(model)
    del layers['lm_head']

    make_quant_kwargs = {
        'module': model,
        'names': layers,
        'bits': 4,
    }
    if 'groupsize' in gptq_args:
        make_quant_kwargs['groupsize'] = 128
    if 'faster' in gptq_args:
        make_quant_kwargs['faster'] = False
    if 'kernel_switch_threshold' in gptq_args:
        make_quant_kwargs['kernel_switch_threshold'] = 128

    make_quant(**make_quant_kwargs)
    del layers
    model.load_state_dict(safe_load(checkpoint_path), strict=False)
    model.seqlen = 2048
    return model
