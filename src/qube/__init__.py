from qube.utils import set_mpl_theme, runtime_env, reload_pyx_module
reload_pyx_module('.')

from loguru import logger
import sys, stackprinter


def formatter(record):
    end = record["extra"].get("end", "\n")
    fmt = "<lvl>{message}</lvl>%s" % end
    if record["level"].name in {"WARNING", "SNAKY"}:
        fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

    prefix = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [ <level>%s</level> ] " % record["level"].icon

    if record["exception"] is not None:
        # stackprinter.set_excepthook(style='darkbg2')
        record["extra"]["stack"] = stackprinter.format(record["exception"], style="darkbg")
        fmt += "\n{extra[stack]}\n"

    if record["level"].name in {"TEXT"}:
        prefix = ""

    return prefix + fmt


config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}"},
    ],
    "extra": {"user": "someone"},
}


logger.configure(**config)
logger.remove(None)
logger.add(sys.stdout, format=formatter, colorize=True)
logger = logger.opt(colors=True)

# registering magic for jupyter notebook
if runtime_env() in ['notebook', 'shell']:
    from IPython.core.magic import (Magics, magics_class, line_magic, line_cell_magic)
    from IPython import get_ipython

    @magics_class
    class QubeMagics(Magics):
        # process data manager
        __manager = None

        @line_magic
        def qubed(self, line: str):
            self.qube_setup('dark')

        @line_magic
        def qubel(self, line: str):
            self.qube_setup('light')

        @line_magic
        def qube_setup(self, line: str):
            """
            QUBE framework initialization
            """
            import os

            tpl_path = os.path.join(os.path.dirname(__file__), "_nb_magic.py")
            # print("TPL:", tpl_path)
            with open(tpl_path, 'r', encoding="utf8") as myfile:
                s = myfile.read()

            exec(s, self.shell.user_ns)

            # setup more funcy mpl theme instead of ugly default
            if line:
                if 'dark' in line.lower():
                    set_mpl_theme('dark')

                elif 'light' in line.lower():
                    set_mpl_theme('light')

            # install additional plotly helpers
            # from qube.charting.plot_helpers import install_plotly_helpers
            # install_plotly_helpers()

        def _get_manager(self):
            if self.__manager is None:
                import multiprocessing as m
                self.__manager = m.Manager()
            return self.__manager

        @line_cell_magic
        def proc(self, line, cell=None):
            """
            Run cell in separate process

            >>> %%proc x, y as MyProc1
            >>> x.set('Hello')
            >>> y.set([1,2,3,4])
            
            """
            import multiprocessing as m
            import time, re

            # create ext args
            name = None
            if line:
                # check if custom process name was provided
                if ' as ' in line:
                    line, name = line.split('as')
                    if not name.isspace():
                        name = name.strip()
                    else:
                        print('>>> Process name must be specified afer "as" keyword !')
                        return

                ipy = get_ipython()
                for a in [x for x in re.split('[\ ,;]', line.strip()) if x]:
                    ipy.push({a: self._get_manager().Value(None, None)})

            # code to run
            lines = '\n'.join(['    %s' % x for x in cell.split('\n')])

            def fn():
                result = get_ipython().run_cell(lines)

                # send errors to parent
                if result.error_before_exec:
                    raise result.error_before_exec

                if result.error_in_exec:
                    raise result.error_in_exec

            t_start = str(time.time()).replace('.', '_')
            f_id = f'proc_{t_start}' if name is None else name
            if self._is_task_name_already_used(f_id):
                f_id = f"{f_id}_{t_start}"

            task = m.Process(target=fn, name=f_id)
            task.start()
            print(' -> Task %s is started' % f_id)

        def _is_task_name_already_used(self, name):
            import multiprocessing as m
            for p in m.active_children():
                if p.name == name:
                    return True
            return False

        @line_magic
        def list_proc(self, line):
            import multiprocessing as m
            for p in m.active_children():
                print(p.name)

        @line_magic
        def kill_proc(self, line):
            import multiprocessing as m
            for p in m.active_children():
                if line and p.name.startswith(line):
                    p.terminate()


    # registering magic here
    get_ipython().register_magics(QubeMagics)
