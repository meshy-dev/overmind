# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Any
import time
import os
import sys
import logging
import importlib

# -- third party --
import rpyc.utils.factory
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler as Pickler

# -- own --
from .utils.misc import hook
from .common import OvermindObjectRef

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
            pass

        if self._is_client_ok():
            return

        log.debug('Starting overmind server as daemon...')
        # if os.system(f'{sys.executable} -m overmind.server --daemon') != 0:
        if os.system('overmind-server --daemon') != 0:
            raise RuntimeError('Failed to start overmind server')

        time.sleep(0.5)

        try:
            log.debug('Connecting to newly spawned overmind server...')
            self.client = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
        except Exception:
            log.exception('Failed to spawn overmind server')

        if self._is_client_ok():
            return

        log.warning('Could not connect to overmind server, falling back to local mode')
        self.enabled = False

    def _convert_to_refs(self, obj):
        if isinstance(obj, (list, tuple)):
            return obj.__class__([self._convert_to_refs(x) for x in obj])
        elif isinstance(obj, dict):
            return {k: self._convert_to_refs(v) for k, v in obj.items()}
        elif (rid := getattr(obj, '_overmind_ref', None)) is not None:
            return OvermindObjectRef(str(rid))
        else:
            return obj

    def load(self, fn, *args, **kwargs):
        self._init_client()

        if not self.enabled:
            return fn(*args, **kwargs)

        payload = self._convert_to_refs((fn, args, kwargs))

        b: bytes = self.client.root.load(bytes(Pickler.dumps(payload)))
        return Pickler.loads(b)


om = OvermindClient()
load = om.load


def monkey_patch(modulename, clsname, method):
    try:
        module = importlib.import_module(modulename)
        if clsname is None:
            cls = module
        else:
            cls = getattr(module, clsname)
    except (ModuleNotFoundError, AttributeError):
        log.info(f'Could not import {modulename}.{clsname}, skipping monkey patching')
        return

    hook(cls, name=method)(load)
    log.info(f'Patched {modulename}.{clsname}.{method}')


def monkey_patch_all():

    from diffusers.models.modeling_utils import ModelMixin

    monkey_patch('diffusers.pipelines.pipeline_utils',   'DiffusionPipeline',       'from_pretrained')
    monkey_patch('diffusers.models.modeling_utils',      'ModelMixin',              'from_pretrained')
    monkey_patch('diffusers.schedulers.scheduler_utils', 'SchedulerMixin',          'from_pretrained')
    monkey_patch('transformers.modeling_utils',          'PreTrainedModel',         'from_pretrained')
    monkey_patch('transformers.tokenization_utils_base', 'PreTrainedTokenizerBase', 'from_pretrained')
    monkey_patch('transformers',                         'AutoProcessor',           'from_pretrained')
    monkey_patch('torchvision.models.vgg',               None,                      'vgg19')
    monkey_patch('torchvision.models.vgg',               None,                      'vgg16')


def _init():
    mp.set_sharing_strategy('file_system')


_init()
