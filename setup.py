# -*- coding: utf-8 -*-
from setuptools import setup

INSTALL_REQUIRES = [
    'daemonize>=2.5.0',
]
ENTRY_POINTS = {
    'console_scripts': [
        'overmind-server = overmind.server:start',
        'overmind-list = overmind.cli:list_loaded',
        'overmind-shutdown = overmind.cli:shutdown',
        'overmind-drop-shell = overmind.cli:drop_shell',
    ],
}

setup_kwargs = {
    'name': 'overmind',
    'version': '0.1.41.dev4+g9e794a9.d20240514',
    'description': 'Daemon to serve shared PyTorch models',
    'url': '',
    'packages': [
        'overmind',
        'overmind.utils',
    ],
    'package_dir': {'': 'src'},
    'package_data': {'': ['*']},
    'install_requires': INSTALL_REQUIRES,
    'python_requires': '>=3.9',
    'entry_points': ENTRY_POINTS,

}
import pickle

context_dump = b'\x80\x04\x95\xd4\x04\x00\x00\x00\x00\x00\x00\x8c\x16pdm.backend.hooks.base\x94\x8c\x07Context\x94\x93\x94)\x81\x94}\x94(\x8c\tbuild_dir\x94\x8c\x07pathlib\x94\x8c\tPosixPath\x94\x93\x94(\x8c\x01/\x94\x8c\x04home\x94\x8c\x06proton\x94\x8c\x03dev\x94\x8c\x08overmind\x94\x8c\n.pdm-build\x94t\x94R\x94\x8c\x08dist_dir\x94h\x08(h\th\nh\x0bh\x0ch\r\x8c\x04dist\x94t\x94R\x94\x8c\x06kwargs\x94}\x94\x8c\x12metadata_directory\x94Ns\x8c\x07builder\x94\x8c\x11pdm.backend.wheel\x94\x8c\x0cWheelBuilder\x94\x93\x94h\x08(h\th\nh\x0bh\x0ch\rt\x94R\x94}\x94\x86\x94R\x94}\x94\x8c\x06config\x94\x8c\x12pdm.backend.config\x94\x8c\x06Config\x94\x93\x94)\x81\x94}\x94(\x8c\x04root\x94h\x08(h\th\nh\x0bh\x0ch\rt\x94R\x94\x8c\x04data\x94}\x94(\x8c\x07project\x94}\x94(\x8c\x04name\x94\x8c\x08overmind\x94\x8c\x07dynamic\x94]\x94\x8c\x0bdescription\x94\x8c%Daemon to serve shared PyTorch models\x94\x8c\x07authors\x94]\x94}\x94(\x8c\x04name\x94\x8c\x06Proton\x94\x8c\x05email\x94\x8c\x10feisuzhu@163.com\x94ua\x8c\x0cdependencies\x94]\x94\x8c\x10daemonize>=2.5.0\x94a\x8c\x0frequires-python\x94\x8c\x05>=3.9\x94\x8c\x06readme\x94\x8c\tREADME.md\x94\x8c\x07license\x94}\x94\x8c\x04text\x94\x8c\x06Apache\x94s\x8c\x07scripts\x94}\x94(\x8c\x0fovermind-server\x94\x8c\x15overmind.server:start\x94\x8c\rovermind-list\x94\x8c\x18overmind.cli:list_loaded\x94\x8c\x11overmind-shutdown\x94\x8c\x15overmind.cli:shutdown\x94\x8c\x13overmind-drop-shell\x94\x8c\x17overmind.cli:drop_shell\x94u\x8c\x07version\x94\x8c\x1e0.1.41.dev4+g9e794a9.d20240514\x94u\x8c\x0cbuild-system\x94}\x94(\x8c\x08requires\x94]\x94\x8c\x0bpdm-backend\x94a\x8c\rbuild-backend\x94\x8c\x0bpdm.backend\x94u\x8c\x04tool\x94}\x94\x8c\x03pdm\x94}\x94(\x8c\x0cdistribution\x94\x88\x8c\x05build\x94}\x94\x8c\x0erun-setuptools\x94\x88s\x8c\x07options\x94}\x94(\x8c\x03add\x94]\x94\x8c\x0e--no-isolation\x94a\x8c\x07install\x94]\x94\x8c\x0e--no-isolation\x94a\x8c\x05build\x94]\x94\x8c\x0e--no-isolation\x94au\x8c\x07version\x94}\x94\x8c\x06source\x94\x8c\x03scm\x94susu\x8c\x08metadata\x94h#\x8c\x08Metadata\x94\x93\x94)\x81\x94}\x94\x8c\x0c_Table__data\x94h.sb\x8c\x0cbuild_config\x94h#\x8c\x0bBuildConfig\x94\x93\x94)\x81\x94}\x94(h(h*hvh`ububsbub.'
context = pickle.loads(context_dump)
builder = context.builder
builder.call_hook("pdm_build_update_setup_kwargs", context, setup_kwargs)


setup(**setup_kwargs)
