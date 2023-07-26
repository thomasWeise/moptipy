"""A tool for writing a section with system information into log files."""
import contextlib
import importlib.metadata as ilm
import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from typing import Final, Iterable, cast

import psutil  # type: ignore

import moptipy.version as ver
from moptipy.api import logging
from moptipy.utils.logger import (
    CSV_SEPARATOR,
    KEY_VALUE_SEPARATOR,
    SCOPE_SEPARATOR,
    InMemoryLogger,
    KeyValueLogSection,
    Logger,
)
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


def __cpu_affinity(proc: psutil.Process | None = None) -> str | None:
    """
    Get the CPU affinity.

    :param proc: the process handle
    :return: the CPU affinity string.
    """
    if proc is None:
        proc = psutil.Process()
    if proc is None:
        return None
    cpua = proc.cpu_affinity()
    if cpua is not None:
        cpua = CSV_SEPARATOR.join(map(str, cpua))
        if len(cpua) > 0:
            return cpua
    return None


#: the dependencies
__DEPENDENCIES: set[str] | None = {
    "contourpy", "cycler", "fonttools", "intel-cmplr-lib-rt", "joblib",
    "kiwisolver", "llvmlite", "matplotlib", "moptipy", "numba", "numpy",
    "packaging", "pdfo", "Pillow", "psutil", "pyparsing", "python-dateutil",
    "scikit-learn", "scipy", "six", "threadpoolctl"}


def is_make_build() -> bool:
    """
    Check if the program was run inside a `make` build.

    :returns: `True` if this process is executed as part of a `make` build
        process, `False` otherwise.

    >>> isinstance(is_make_build(), bool)
    True
    >>> ns = lambda prc: False if prc is None else (  # noqa: E731
    ...     "make" in prc.name() or ns(prc.parent()))
    >>> is_make_build() == ns(psutil.Process(os.getppid()))
    True
    """
    obj: Final[object] = is_make_build
    key: Final[str] = "_value"
    if hasattr(obj, key):
        return cast(bool, getattr(obj, key))

    ret: bool = False
    with contextlib.suppress(Exception):
        process: psutil.Process = psutil.Process(os.getppid())
        while process is not None:
            name = process.cmdline()[0]
            if not isinstance(name, str):
                break
            name = os.path.basename(name)
            if (name == "make") or (name.startswith("make.")):
                ret = True
                break
            process = process.parent()

    setattr(obj, key, ret)
    return ret


def add_dependency(dependency: str,
                   ignore_if_make_build: bool = False) -> None:
    """
    Add a dependency so that its version can be stored in log files.

    Warning: You must add all dependencies *before* the first log file is
    written. As soon as the :func:`log_sys_info` is invoked for the first
    time, adding new dependencies will cause an error. And writing a log
    file via the :mod:`~moptipy.api.experiment` API or to a file specified in
    the :mod:`~moptipy.api.execution` API will invoke this function.

    :param dependency: the basic name of the library, exactly as you would
        `import` it in a Python module. For example, to include the version of
        `numpy` in the log files, you would do `add_dependency("numpy")` (of
        course, the version of `numpy` is already automatically included
        anyway).
    :param ignore_if_make_build: should this dependency be ignored if this
        method is invoked during a `make` build? This makes sense if the
        dependency itself is a package which is uninstalled and then
        re-installed during a `make` build process. In such a situation, the
        dependency version may be unavailable and cause an exception.
    :raises TypeError: if `dependency` is not a string
    :raises ValueError: if `dependency` is an invalid string or the log
        information has already been accessed before and modifying it now is
        not permissible.
    """
    if not isinstance(dependency, str):
        raise type_error(dependency, "dependency", str)
    if (len(dependency) <= 0) or (dependency != dependency.strip())\
            or (" " in dependency):
        raise ValueError(f"Invalid dependency string {dependency!r}.")
    if __DEPENDENCIES is None:
        raise ValueError(
            f"Too late. Cannot add dependency {dependency!r} anymore.")
    if ignore_if_make_build and is_make_build():
        return
    __DEPENDENCIES.add(dependency)


# noinspection PyBroadException
def __make_sys_info() -> str:
    """
    Build the system info string.

    This method is only used once and then deleted.

    :returns: the system info string.
    """
    global __DEPENDENCIES  # noqa: PLW0603  # pylint: disable=W0603
    if __DEPENDENCIES is None:
        raise ValueError("Cannot re-create log info.")
    dep: Final[set[str]] = __DEPENDENCIES
    __DEPENDENCIES = None  # noqa: PLW0603  # pylint: disable=W0603

    def __v(sec: KeyValueLogSection, key: str, value) -> None:
        """
        Create a key-value pair if value is not empty.

        :param sec: the section to write to.
        :param key: the key
        :param value: an arbitrary value, maybe consisting of multiple lines.
        """
        if value is None:
            return
        value = " ".join([ts.strip() for ts in
                          str(value).strip().split("\n")]).strip()
        if len(value) <= 0:
            return
        sec.key_value(key, value)

    def __get_processor_name() -> str | None:
        """
        Get the processor name.

        :returns: a string if there is any processor information
        """
        with contextlib.suppress(Exception):
            if platform.system() == "Windows":
                return platform.processor()
            if platform.system() == "Linux":
                for line in Path.path("/proc/cpuinfo").read_all_list():
                    if "model name" in line:
                        return re.sub(pattern=".*model name.*:",
                                      repl="", string=line, count=1).strip()
        return None

    def __get_mem_size_sysconf() -> int | None:
        """
        Get the memory size information from sysconf.

        :returns: an integer with the memory size if available
        """
        with contextlib.suppress(Exception):
            k1 = "SC_PAGESIZE"
            if k1 not in os.sysconf_names:
                k1 = "SC_PAGE_SIZE"
                if k1 not in os.sysconf_names:
                    return None
            v1 = os.sysconf(k1)
            if not isinstance(v1, int):
                return None
            k2 = "SC_PHYS_PAGES"
            if k2 not in os.sysconf_names:
                k2 = "_SC_PHYS_PAGES"
                if k2 not in os.sysconf_names:
                    return None
            v2 = os.sysconf(k2)
            if not isinstance(v1, int):
                return None

            if (v1 > 0) and (v2 > 0):
                return v1 * v2
        return None

    def __get_mem_size_meminfo() -> int | None:
        """
        Get the memory size information from meminfo.

        :returns: an integer with the memory size if available
        """
        with contextlib.suppress(Exception):
            meminfo = {i.split()[0].rstrip(":"): int(i.split()[1])
                       for i in Path.path("/proc/meminfo").read_all_list()}
            mem_kib = meminfo["MemTotal"]  # e.g. 3921852
            mem_kib = int(mem_kib)
            if mem_kib > 0:
                return 1024 * mem_kib
        return None

    def __get_mem_size() -> int | None:
        """
        Get the memory size information from any available source.

        :returns: an integer with the memory size if available
        """
        vs = __get_mem_size_sysconf()
        if vs is None:
            vs = __get_mem_size_meminfo()
        if vs is None:
            return psutil.virtual_memory().total
        return vs

    # log all information in memory to convert it to one constant string.
    with InMemoryLogger() as imr:
        with imr.key_values(logging.SECTION_SYS_INFO) as kv:
            with kv.scope(logging.SCOPE_SESSION) as k:
                __v(k, logging.KEY_SESSION_START,
                    datetime.now(tz=timezone.utc))
                __v(k, logging.KEY_NODE_NAME, platform.node())
                proc = psutil.Process()
                __v(k, logging.KEY_PROCESS_ID, hex(proc.pid))
                cpua = __cpu_affinity(proc)
                if cpua is not None:
                    __v(k, logging.KEY_CPU_AFFINITY, cpua)
                del proc, cpua

                # get the command line and working directory of the process
                with contextlib.suppress(Exception):
                    proc = psutil.Process(os.getpid())
                    cmd = proc.cmdline()
                    if isinstance(cmd, Iterable):
                        __v(k, logging.KEY_COMMAND_LINE,
                            " ".join(map(repr, proc.cmdline())))
                    cwd = proc.cwd()
                    if isinstance(cwd, str):
                        __v(k, logging.KEY_WORKING_DIRECTORY,
                            repr(proc.cwd()))

                # see https://stackoverflow.com/questions/166506/.
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # doesn't even have to be reachable
                    s.connect(("10.255.255.255", 1))
                    ip = s.getsockname()[0]
                except Exception:
                    ip = "127.0.0.1"
                finally:
                    s.close()
                __v(k, logging.KEY_NODE_IP, ip)

            with kv.scope(logging.SCOPE_VERSIONS) as k:
                for package in sorted(dep):
                    if package == "moptipy":
                        __v(k, "moptipy", ver.__version__)
                    else:
                        __v(k, package.replace("-", ""),
                            ilm.version(package).strip())

            with kv.scope(logging.SCOPE_HARDWARE) as k:
                __v(k, logging.KEY_HW_MACHINE, platform.machine())
                __v(k, logging.KEY_HW_N_PHYSICAL_CPUS,
                    psutil.cpu_count(logical=False))
                __v(k, logging.KEY_HW_N_LOGICAL_CPUS,
                    psutil.cpu_count(logical=True))

                # store the CPU speed information
                cpuf: dict[tuple[int, int], int] = {}
                total: int = 0
                for cf in psutil.cpu_freq(True):
                    t = (int(cf.min), int(cf.max))
                    cpuf[t] = cpuf.get(t, 0) + 1
                    total += 1
                memlst: list[tuple[int, ...]]
                if total > 1:
                    memlst = [(key[0], key[1], value) for
                              key, value in cpuf.items()]
                    memlst.sort()
                else:
                    memlst = list(cpuf)

                def __make_mhz_str(tpl: tuple[int, ...]) -> str:
                    """Convert a MHz tuple to a string."""
                    base: str = f"({tpl[0]}MHz..{tpl[1]}MHz)" \
                        if tpl[1] > tpl[0] else f"{tpl[0]}MHz"
                    return base if (len(tpl) < 3) or (tpl[2] <= 1) \
                        else f"{base}*{tpl[2]}"
                __v(k, logging.KEY_HW_CPU_MHZ,
                    "+".join([__make_mhz_str(t) for t in memlst]))

                __v(k, logging.KEY_HW_BYTE_ORDER, sys.byteorder)
                __v(k, logging.KEY_HW_CPU_NAME, __get_processor_name())
                __v(k, logging.KEY_HW_MEM_SIZE, __get_mem_size())

            with kv.scope(logging.SCOPE_PYTHON) as k:
                __v(k, logging.KEY_PYTHON_VERSION, sys.version)
                __v(k, logging.KEY_PYTHON_IMPLEMENTATION,
                    platform.python_implementation())

            with kv.scope(logging.SCOPE_OS) as k:
                __v(k, logging.KEY_OS_NAME, platform.system())
                __v(k, logging.KEY_OS_RELEASE, platform.release())
                __v(k, logging.KEY_OS_VERSION, platform.version())

        lst = imr.get_log()

    if len(lst) < 3:
        raise ValueError("sys info turned out to be empty?")
    __DEPENDENCIES = None
    return "\n".join(lst[1:(len(lst) - 1)])


def get_sys_info() -> str:
    r"""
    Get the system information as string.

    :returns: the system information as string

    >>> raw_infos = get_sys_info()
    >>> raw_infos is get_sys_info()  # caching!
    True
    >>> for k in raw_infos.split("\n"):
    ...     print(k[:k.find(": ")])
    session.start
    session.node
    session.processId
    session.cpuAffinity
    session.commandLine
    session.workingDirectory
    session.ipAddress
    version.Pillow
    version.contourpy
    version.cycler
    version.fonttools
    version.intelcmplrlibrt
    version.joblib
    version.kiwisolver
    version.llvmlite
    version.matplotlib
    version.moptipy
    version.numba
    version.numpy
    version.packaging
    version.pdfo
    version.psutil
    version.pyparsing
    version.pythondateutil
    version.scikitlearn
    version.scipy
    version.six
    version.threadpoolctl
    hardware.machine
    hardware.nPhysicalCpus
    hardware.nLogicalCpus
    hardware.cpuMhz
    hardware.byteOrder
    hardware.cpu
    hardware.memSize
    python.version
    python.implementation
    os.name
    os.release
    os.version
    """
    the_object: Final[object] = get_sys_info
    the_attr: Final[str] = "__the_sysinfo"
    if hasattr(the_object, the_attr):
        return getattr(the_object, the_attr)
    sys_info: Final[str] = __make_sys_info()
    setattr(the_object, the_attr, sys_info)
    return sys_info


def update_sys_info_cpu_affinity() -> None:
    """Update the CPU affinity of the system information."""
    sys_info_str = get_sys_info()
    start = (f"\n{logging.SCOPE_SESSION}{SCOPE_SEPARATOR}"
             f"{logging.KEY_CPU_AFFINITY}{KEY_VALUE_SEPARATOR}")
    start_i = sys_info_str.find(start)
    if start_i < 0:
        return   # no affinity, don't need to update
    start_i += len(start)
    end_i = sys_info_str.find("\n", start_i)
    if end_i <= start_i:
        raise ValueError(f"Empty {logging.KEY_CPU_AFFINITY}?")
    affinity = __cpu_affinity()
    if affinity is None:
        raise ValueError(
            f"first affinity query is {sys_info_str[start_i:end_i]},"
            f" but second one is None?")
    sys_info_str = f"{sys_info_str[:start_i]}{affinity}{sys_info_str[end_i:]}"

    the_object: Final[object] = get_sys_info
    the_attr: Final[str] = "__the_sysinfo"
    setattr(the_object, the_attr, sys_info_str)


def log_sys_info(logger: Logger) -> None:
    r"""
    Write the system information section to a logger.

    The concept of this method is that we only construct the whole system
    configuration exactly once in memory and then always directly flush it
    as a string to the logger. This is much more efficient than querying it
    every single time.

    :param logger: the logger

    >>> from moptipy.utils.logger import InMemoryLogger
    >>> with InMemoryLogger() as l:
    ...     log_sys_info(l)
    ...     log = l.get_log()
    >>> print(log[0])
    BEGIN_SYS_INFO
    >>> print(log[-1])
    END_SYS_INFO
    >>> log[1:-1] == get_sys_info().split('\n')
    True
    """
    with logger.text(logging.SECTION_SYS_INFO) as txt:
        txt.write(get_sys_info())
