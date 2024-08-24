# Cremage note: This is the engine file that I need to tweak.
# Refactored streamlit_helpers.py
import os
import sys
import logging
import torch
import gc

lowvram_mode = False

def load_model(model):
    # model.cuda()
    model.to(os.environ.get("GPU_DEVICE", "cpu"))
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Cremage added
    gc.collect()


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        # model.cuda()
        model.to(os.environ.get("GPU_DEVICE", "cpu"))
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Cremage added
    return model

def initial_svd_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        # model.half()
        pass
    else:
        model.to(os.environ.get("GPU_DEVICE", "cpu"))
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Cremage added
    return model

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Cremage added
