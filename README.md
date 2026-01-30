# Overmind

**Cut PyTorch model loading time from 15s to 0.2s with zero-copy shared memory caching.**

[![PyPI version](https://badge.fury.io/py/overmind-cache.svg)](https://badge.fury.io/py/overmind-cache)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Overmind is a non-intrusive caching library that dramatically speeds up PyTorch model loading by storing serialized models in shared memory. Once a model is loaded, subsequent loads from any process take milliseconds instead of seconds.

Named after the [Overmind from StarCraft](https://starcraft.fandom.com/wiki/Overmind), it coordinates model caching across processes like the Overmind coordinates the Zerg Swarm.

Note that the package name on PyPI is `overmind-cache`, since `overmind` is taken.

## Features

- **Fast model loading** - First load caches to shared memory; subsequent loads are ~5x faster
- **Process-agnostic** - Cache persists across process restarts via a background server
- **Non-intrusive** - Just add one line of code; no changes to model loading logic
- **Memory efficient** - Multiple processes share the same cached tensors in memory
- **Broad compatibility** - Works with `diffusers`, `transformers`, `bitsandbytes` quantization, and vanilla `torch.load`

## Installation

```bash
pip install overmind-cache
```

Or install from source:

```bash
git clone https://github.com/taichi-dev/overmind.git
cd overmind
pip install -e .
```

## Quick Start

### Option 1: Monkey Patching (Recommended)

Add a single line at the top of your script to automatically accelerate all supported model loading:

```python
import overmind.api
overmind.api.monkey_patch_all()

# Your existing code works unchanged!
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipeline.to('cuda')
# First run: ~24s
# Subsequent runs: ~1s (mainly consumed by .to('cuda'))
```

### Option 2: Explicit API

For the ones don't like monkey-patching, use the `load` function directly:

```python
from overmind.api import load
from diffusers import DiffusionPipeline

pipeline = load(
    DiffusionPipeline.from_pretrained,
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
```

## Supported Libraries

Overmind automatically patches these loading functions:

| Library          | Functions                                                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **Diffusers**    | `DiffusionPipeline.from_pretrained`, `ModelMixin.from_pretrained`, `SchedulerMixin.from_pretrained`, `FromSingleFileMixin.from_single_file` |
| **Transformers** | `PreTrainedModel.from_pretrained`, `PreTrainedTokenizerBase.from_pretrained`, `AutoProcessor.from_pretrained`, `pipeline`                   |
| **PyTorch**      | `torch.load`, `torch.jit.load`                                                                                                              |
| **Safetensors**  | `safetensors.torch.load_file`                                                                                                               |
| **TorchVision**  | `vgg16`, `vgg19`                                                                                                                            |
| **OpenCLIP**     | `create_model_and_transforms`                                                                                                               |

### Custom Patch Points

Create an `overmind.cfg` file in your package root to add custom patch points:

```
# overmind.cfg
mylib.models::MyModel.from_pretrained
mylib.utils::load_checkpoint
```

## CLI Commands

```bash
# Start the server manually (usually auto-started)
overmind-server

# Start as daemon
overmind-server --daemon

# List cached models
overmind-list

# Shutdown the server (clears cache)
overmind-shutdown
```

## Environment Variables

| Variable                  | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| `OVERMIND_DISABLE`        | Set to any value to disable Overmind, falling back to a local cache |
| `OVERMIND_NO_LOCAL_CACHE` | Disable local caching too                                           |


## Benchmarks

Loading a Stable Diffusion ControlNet pipeline with VAE, on Linux + Intel i9-11900K + RTX 4090:

Using [demo-vae.py](demo-vae.py) as example:


| Run                 | `vae` | `depth` | `edge` | `pipeline` | to('cuda') | Total  |
|---------------------|-------|---------|--------|------------|------------|--------|
| w/o Overmind (2nd+) | 1.18s | 0.98s   | 1.41s  | 1.65s      | 0.91s      | 6.16s  |
| w/ Overmind (1st)   | 5.44s | 5.17s   | 5.41s  | 7.29s      | 0.86s      | 24.20s |
| w/ Overmind (2nd+)  | 0.00s | 0.01s   | 0.01s  | 0.20s      | 0.87s      | 1.12s  |

The first load with Overmind is slower due to pickling overhead. Subsequent loads are **5-6x faster** than without Overmind, with the only remaining cost being the `to('cuda')` transfer.

## License

Apache 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Developed by [Taichi Graphics](https://github.com/taichi-dev) for production AI inference workloads.
