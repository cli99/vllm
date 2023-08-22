import torch
import time

def set_model_stage(model, stage):
    model.stage = stage

def add_model_hooks(model):
    def start_time_hook(module, input):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__start_time__ = time.time()

    def end_time_hook(module, input, output):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__duration__ = time.time() - module.__start_time__
            module.stage = "decode"

    if not hasattr(model, '__start_time_hook_handle'):
        model.__start_time_hook_handle__ = model.register_forward_pre_hook(
            start_time_hook,
        )

    if not hasattr(model, '__end_time_hook_handle__'):
        model.__end_time_hook_handle__ = model.register_forward_hook(
            end_time_hook,
        )

def remove_model_hooks(module):
        if hasattr(module, '__start_time_hook_handle__'):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, '__end_time_hook_handle__'):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, 'stage'):
            del module.stage
        if hasattr(module, '__duration__'):
            del module.__duration__