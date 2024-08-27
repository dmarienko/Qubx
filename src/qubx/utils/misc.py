from typing import Dict, Optional, Union, List
import os
from collections import OrderedDict, defaultdict, namedtuple
from os.path import basename, exists, dirname, join, expanduser
import time
from pathlib import Path
import joblib
from tqdm.auto import tqdm


def version() -> str:
    # - check current version
    version = "Dev"
    try:
        import importlib_metadata

        version = importlib_metadata.version("qubx")
    except:
        pass

    return version


def install_pyx_recompiler_for_dev():
    from ._pyxreloader import pyx_install_loader

    # if version().lower() == 'dev':
    print(f" >  [{green('dev')}] {red('installing cython rebuilding hook')}")
    pyx_install_loader(["qubx.core", "qubx.ta", "qubx.data", "qubx.strategies"])


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
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            return "notebook"
        elif shell.endswith("TerminalInteractiveShell"):  # Terminal running IPython
            return "shell"
        else:
            return "unknown"  # Other type (?)
    except (NameError, ImportError):
        return "python"  # Probably standard Python interpreter


_QUBX_FLDR = None


def get_local_qubx_folder() -> str:
    global _QUBX_FLDR

    if _QUBX_FLDR is None:
        _QUBX_FLDR = makedirs(os.getenv("QUBXSTORAGE", os.path.expanduser("~/.qubx")))

    return _QUBX_FLDR


def add_project_to_system_path(project_folder: str = "~/projects"):
    """
    Add path to projects folder to system python path to be able importing any modules from project
    from test.Models.handy_utils import some_module
    """
    import sys
    from os.path import expanduser, relpath
    from pathlib import Path

    # we want to track folders with these files as separate paths
    toml = Path("pyproject.toml")
    src = Path("src")

    try:
        prj = Path(relpath(expanduser(project_folder)))
    except ValueError as e:
        # This error can occur on Windows if user folder and python file are on different drives
        print(f"Qube> Error during get path to projects folder:\n{e}")
    else:
        insert_path_iff = lambda p: (sys.path.insert(0, p.as_posix()) if p.as_posix() not in sys.path else None)
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
            print(f"Qube> Cant find {project_folder} folder for adding to python path !")


def is_localhost(host):
    return host.lower() == "localhost" or host == "127.0.0.1"


def __wrap_with_color(code):
    def inner(text, bold=False):
        c = code
        if bold:
            c = "1;%s" % c
        return "\033[%sm%s\033[0m" % (c, text)

    return inner


red, green, yellow, blue, magenta, cyan, white = (
    __wrap_with_color("31"),
    __wrap_with_color("32"),
    __wrap_with_color("33"),
    __wrap_with_color("34"),
    __wrap_with_color("35"),
    __wrap_with_color("36"),
    __wrap_with_color("37"),
)


def logo():
    """
    Some fancy Qubx logo
    """
    print(
        f"""
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀   
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  {red("QUBX")} | {cyan("Quantitative Backtesting Environment")} 
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2024, ver. {magenta(version().rstrip())}
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀ 
        """
    )


class Struct:
    """
    Dynamic structure (similar to matlab's struct it allows to add new properties dynamically)

    >>> a = Struct(x=1, y=2)
    >>> a.z = 'Hello'
    >>> print(a)

    Struct(x=1, y=2, z='Hello')

    >>> Struct(a=234, b=Struct(c=222)).to_dict()

    {'a': 234, 'b': {'c': 222}}

    >>> Struct({'a': 555}, a=123, b=Struct(c=222)).to_dict()

    {'a': 123, 'b': {'c': 222}}
    """

    def __init__(self, *args, **kwargs):
        _odw = OrderedDict(**kwargs)
        if args:
            if isinstance(args[0], dict):
                _odw = OrderedDict(Struct.dict2struct(args[0]).to_dict()) | _odw
            elif isinstance(args[0], Struct):
                _odw = args[0].to_dict() | _odw
        self.__initialize(_odw.keys(), _odw.values())

    def __initialize(self, fields, values):
        self._fields = list(fields)
        self._meta = namedtuple("Struct", " ".join(fields))
        self._inst = self._meta(*values)

    def fields(self) -> list:
        return self._fields

    def __getitem__(self, idx: int):
        return getattr(self._inst, self._fields[idx])

    def __getattr__(self, k):
        return getattr(self._inst, k)

    def __or__(self, other: Union[dict, "Struct"]):
        if isinstance(other, dict):
            other = Struct.dict2struct(other)
        elif not isinstance(other, Struct):
            raise ValueError(f"Can't union with object of {type(other)} type ")
        for f in other.fields():
            self.__setattr__(f, other.__getattr__(f))
        return self

    def __dir__(self):
        return self._fields

    def __repr__(self):
        return self._inst.__repr__()

    def __setattr__(self, k, v):
        if k not in ["_inst", "_meta", "_fields"]:
            new_vals = {**self._inst._asdict(), **{k: v}}
            self.__initialize(new_vals.keys(), new_vals.values())
        else:
            super().__setattr__(k, v)

    def __getstate__(self):
        return self._inst._asdict()

    def __setstate__(self, state):
        self.__init__(**state)

    def __ms2d(self, m) -> dict:
        r = {}
        for f in m._fields:
            v = m.__getattr__(f)
            r[f] = self.__ms2d(v) if isinstance(v, Struct) else v
        return r

    def to_dict(self) -> dict:
        """
        Return this structure as dictionary
        """
        return self.__ms2d(self)

    def copy(self) -> "Struct":
        """
        Returns copy of this structure
        """
        return Struct(self.to_dict())

    @staticmethod
    def dict2struct(d: dict) -> "Struct":
        """
        Convert dictionary to structure
        >>> s = dict2struct({'f_1_0': 1, 'z': {'x': 1, 'y': 2}})
        >>> print(s.z.x)
        1
        """
        m = Struct()
        for k, v in d.items():
            # skip if key is not valid identifier
            if not k.isidentifier():
                print(f"Struct> {k} doesn't look like as identifier - skip it")
                continue
            if isinstance(v, dict):
                v = Struct.dict2struct(v)
            m.__setattr__(k, v)
        return m


def makedirs(path: str, *args) -> str:
    path = os.path.expanduser(os.path.join(*[path, *args]))
    if not exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Stopwatch:
    """
    Stopwatch timer for performance
    """

    starts: Dict[str | None, int] = {}
    counts: Dict[str | None, int] = defaultdict(lambda: 0)
    latencies: Dict[str | None, int] = {}

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Stopwatch, cls).__new__(cls)
        return cls.instance

    def start(self, scope: str | None):
        self.starts[scope] = time.perf_counter_ns()
        self.counts[scope] += 1

    def stop(self, scope: str | None = None) -> int | None:
        t = time.perf_counter_ns()
        s = self.starts.get(scope, None)
        lat = None
        if s:
            lat = t - s
            n = self.counts[scope]
            self.latencies[scope] = (self.latencies.get(scope, lat) * (n - 1) + lat) // n
            del self.starts[scope]
        return lat

    def latency_sec(self, scope: str | None) -> float:
        return self.latencies.get(scope, 0) / 1e9

    def watch(self, scope="global"):
        def _decorator(func):
            info = scope + "." + func.__name__

            def wrapper(*args, **kwargs):
                self.start(info)
                output = func(*args, **kwargs)
                self.stop(info)
                return output

            return wrapper

        return _decorator

    def reset(self):
        self.starts.clear()
        self.counts.clear()
        self.latencies.clear()

    def __str__(self) -> str:
        r = ""
        for l in self.latencies.keys():
            r += f"\n\t<w>{l}</w> took <r>{self.latency_sec(l):.7f}</r> secs"
        return r


def quotify(sx: Union[str, List[str]], quote="USDT"):
    """
    Make XXX<quote> from anything if that anything doesn't end with <quote>
    """
    if isinstance(sx, str):
        return (sx if sx.endswith(quote) else sx + quote).upper()
    elif isinstance(sx, (list, set, tuple)):
        return [quotify(s, quote) for s in sx]
    raise ValueError("Can't process input data !")


def dequotify(sx: Union[str, List[str]], quote="USDT"):
    """
    Turns XXX<quote> to XXX (reverse of quotify)
    """
    if isinstance(sx, str):
        quote = quote.upper()
        if (s := sx.upper()).endswith(quote):
            s = s.split(":")[1] if ":" in s else s  # remove exch: if presented
            return s.split(quote)[0]
    elif isinstance(sx, (list, set, tuple)):
        return [dequotify(s, quote) for s in sx]

    raise ValueError("Can't process input data !")


class ProgressParallel(joblib.Parallel):
    def __init__(self, *args, **kwargs):
        self.total = kwargs.pop("total", None)
        self.silent = kwargs.pop("silent", False)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.silent:
            return joblib.Parallel.__call__(self, *args, **kwargs)
        with tqdm(total=self.total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.silent:
            return
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
