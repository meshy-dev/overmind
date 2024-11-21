import logging
logging.basicConfig(level=logging.DEBUG)

import overmind.api
overmind.api.monkey_patch_all()

from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = overmind.api.load(load_pretrained_model,
    "liuhaotian/llava-v1.6-mistral-7b",
    None,
    "llava-v1.6-mistral-7b",
    load_4bit=True,  # load in 4 bits
)

import time
time.sleep(1000)
