from os.path import basename, exists, dirname, join, expanduser
import glob
from typing import Optional


def pyx_reload(path: str):
    """
    Reload specified cython module
    path must have .pyx extension
    """
    if exists(path):
        f_name, f_ext = basename(path).split('.')
        if f_ext == 'pyx':
            import numpy as np
            import pyximport
            pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True, language_level=3)
            pyximport.load_module(f_name, path, language_level=3, pyxbuild_dir=expanduser("~/.pyxbld"))
            print(" > Reloaded %s" % path)
    else:
        raise ValueError("Path '%s' not found !" % path)


def reload_pyx_module(module_dir: Optional[str]=None):
    _module_dir = dirname(__file__) if module_dir is None else module_dir
    for _m in glob.glob(join(_module_dir, '*.pyx')):
        pyx_reload(_m)