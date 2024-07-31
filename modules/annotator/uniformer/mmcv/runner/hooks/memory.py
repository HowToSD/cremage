# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .hook import HOOKS, Hook


@HOOKS.register_module()
class EmptyCacheHook(Hook):

    def __init__(self, before_epoch=False, after_epoch=True, after_iter=False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, runner):
        if self._after_iter:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def before_epoch(self, runner):
        if self._before_epoch:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def after_epoch(self, runner):
        if self._after_epoch:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
