import diffusers
import torch
import overmind.api

# diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2', custom_pipeline='sudo-ai/zero123plus-pipeline', torch_dtype=torch.float16, local_files_only=False)

overmind.api.load(diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained, 'sudo-ai/zero123plus-v1.2', custom_pipeline='sudo-ai/zero123plus-pipeline', torch_dtype=torch.float16, local_files_only=False)
