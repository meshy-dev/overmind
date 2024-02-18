# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Any
import time
import os
import pickle
import logging
import importlib

# -- third party --
import rpyc.utils.factory
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler as Pickler

# -- own --
from .utils.misc import hook

# -- code --
log = logging.getLogger('overmind.api')


class OvermindClient:

    def __init__(self):
        self.client: Any = None
        self.enabled = True

    def _is_client_ok(self):
        if not self.client:
            return False

        try:
            return self.client.root.ping() == 'pong'
        except Exception:
            return False

    def _init_client(self):
        if not self.enabled:
            return

        if self._is_client_ok():
            return

        try:
            log.debug('Connecting to existing overmind server...')
            self.client = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
        except Exception:
            raise

        if self._is_client_ok():
            return

        log.debug('Starting overmind server as daemon...')
        if os.system('overmind-server --daemon') != 0:
            raise RuntimeError('Failed to start overmind server')

        time.sleep(0.5)

        try:
            log.debug('Connecting to newly spawned overmind server...')
            self.client = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
        except Exception:
            raise

        if self._is_client_ok():
            return

        log.warning('Could not connect to overmind server, falling back to local mode')
        self.enabled = False

    def load(self, fn, *args, **kwargs):
        self._init_client()

        if not self.enabled:
            return fn(*args, **kwargs)

        b: bytes = self.client.root.load(pickle.dumps((fn, args, kwargs)))
        return Pickler.loads(b)


om = OvermindClient()
load = om.load


def monkey_patch(modulename, clsname, method):
    try:
        module = importlib.import_module(modulename)
        cls = getattr(module, clsname)
    except ModuleNotFoundError, AttributeError:
        log.info(f'Could not import {modulename}.{clsname}, skipping monkey patching')
        return

    hook(cls, name=method)(load)
    log.info(f'Patched {modulename}.{clsname}.{method}')


def monkey_patch_all():
    monkey_patch('diffusers.pipelines.pipeline_utils',   'DiffusionPipeline',       'from_pretrained')
    monkey_patch('diffusers.models.modeling_utils',      'ModelMixin',              'from_pretrained')
    monkey_patch('diffusers.schedulers.scheduler_utils', 'SchedulerMixin',          'from_pretrained')
    monkey_patch('transformers.modeling_utils',          'PreTrainedModel',         'from_pretrained')
    monkey_patch('transformers.tokenization_utils_base', 'PreTrainedTokenizerBase', 'from_pretrained')
    monkey_patch('transformers',                         'AutoProcessor',           'from_pretrained')


def _init():
    mp.set_sharing_strategy('file_system')


_init()
