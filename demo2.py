import overmind.api
overmind.api.monkey_patch_all()

import meshflow.driver
dream = meshflow.driver._load_imagedream('cpu', True)

import os
if os.environ.get('EVAL') == '1':
    import IPython
    IPython.embed()
