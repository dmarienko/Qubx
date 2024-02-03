from os.path import basename, exists, dirname, join, expanduser
import glob
from typing import Optional
from pathlib import Path


def version() -> str:
    # - check current version
    version = 'Dev'
    try: 
        import importlib_metadata
        version = importlib_metadata.version('qube2')
    except:
        pass

    return version


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
            if version().lower() == 'dev':
                print(f"\t{green('>>>')} [{green('dev')}] : module {blue(f_name)} reloaded")
    else:
        raise ValueError("Path '%s' not found !" % path)


def reload_pyx_module(module_dir: Optional[str]=None):
    from os.path import abspath
    _module_dir = abspath(dirname(__file__) if module_dir is None else module_dir)
    # print(abspath(_module_dir))
    for f in Path(_module_dir).iterdir():
        if f.suffix == '.pyx':
            pyx_reload(str(f))
        if f.is_dir():
            for _m in glob.glob(join(f, '*.pyx')):
                pyx_reload(_m)

    # for _m in glob.glob(join(_module_dir, '*.pyx')):
        # pyx_reload(_m)


def runtime_env():
    """
    Check what environment this script is being run under
    :return: environment name, possible values:
             - 'notebook' jupyter notebook
             - 'shell' any interactive shell (ipython, PyCharm's console etc)
             - 'python' standard python interpreter
             - 'unknown' can't recognize environment
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__

        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return 'notebook'
        elif shell.endswith('TerminalInteractiveShell'):  # Terminal running IPython
            return 'shell'
        else:
            return 'unknown'  # Other type (?)
    except (NameError, ImportError):
        return 'python'  # Probably standard Python interpreter


def add_project_to_system_path(project_folder:str = '~/projects'):
    """
    Add path to projects folder to system python path to be able importing any modules from project
    from test.Models.handy_utils import some_module
    """
    import sys
    from os.path import expanduser, relpath
    from pathlib import Path
    
    # we want to track folders with these files as separate paths
    toml = Path('pyproject.toml')
    src = Path('src')
    
    try:
        prj = Path(relpath(expanduser(project_folder)))
    except ValueError as e:
        # This error can occur on Windows if user folder and python file are on different drives
        print(f"Qube> Error during get path to projects folder:\n{e}")
    else:
        insert_path_iff = lambda p: sys.path.insert(0, p.as_posix()) if p.as_posix() not in sys.path else None
        if prj.exists():
            insert_path_iff(prj)
            
            for di in prj.iterdir():
                _src = di / src
                if (di / toml).exists():
                    # when we have src/
                    if _src.exists() and _src.is_dir():
                        insert_path_iff(_src)
                    else:
                        insert_path_iff(di)
        else:
            print(f'Qube> Cant find {project_folder} folder for adding to python path !')


def is_localhost(host):
    return host.lower() == 'localhost' or host == '127.0.0.1'


def __wrap_with_color(code):
    def inner(text, bold=False):
        c = code
        if bold:
            c = "1;%s" % c
        return "\033[%sm%s\033[0m" % (c, text)

    return inner


red, green, yellow, blue, magenta, cyan, white = (
    __wrap_with_color('31'),
    __wrap_with_color('32'),
    __wrap_with_color('33'),
    __wrap_with_color('34'),
    __wrap_with_color('35'),
    __wrap_with_color('36'),
    __wrap_with_color('37'),
)
