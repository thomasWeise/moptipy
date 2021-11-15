"""A tool for writing a section with system information into log files."""
import importlib.metadata as ilm
import os
import platform
import re
import sys
from datetime import datetime
from typing import Optional, Final

import moptipy.version as ver
from moptipy.utils import logging
from moptipy.utils.logger import InMemoryLogger, Logger, KeyValueSection
from moptipy.utils.path import Path


def __make_sys_info() -> str:
    """
    Build the system info string.

    This method is only used once and then deleted.

    :returns: the system info string.
    :rtype: str
    """
    def __v(sec: KeyValueSection, key: str, value) -> None:
        """
        Create a key-value pair if value is not empty.

        :param KeyValueSection sec: the section to write to.
        :param str key: the key
        :param value: an arbitrary value, maybe consisting of multiple lines.
        """
        if value is None:
            return
        value = " ".join([s.strip() for s in
                          str(value).strip().split("\n")]).strip()
        if len(value) <= 0:
            return
        sec.key_value(key, value)

    # noinspection PyBroadException
    def __get_processor_name() -> Optional[str]:
        """
        Get the processor name.

        :returns: a string if there is any processor information
        :rtype: Optional[str]
        """
        try:
            if platform.system() == "Windows":
                return platform.processor()
            if platform.system() == "Linux":
                for line in Path.path("/proc/cpuinfo").read_all_list():
                    if "model name" in line:
                        return re.sub(".*model name.*:", "", line, 1).strip()
        except BaseException:
            pass
        return None

    # noinspection PyBroadException
    def __get_mem_size_sysconf() -> Optional[int]:
        """
        Get the memory size information from sysconf.

        :returns: an integer with the memory size if available
        :rtype: Optional[int]
        """
        try:
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

        except BaseException:
            pass
        return None

    # noinspection PyBroadException
    def __get_mem_size_meminfo() -> Optional[int]:
        """
        Get the memory size information from meminfo.

        :returns: an integer with the memory size if available
        :rtype: Optional[int]
        """
        try:
            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1]))
                           for i in Path.path('/proc/meminfo').read_all_list())
            mem_kib = meminfo['MemTotal']  # e.g. 3921852
            mem_kib = int(mem_kib)
            if mem_kib > 0:
                return 1024 * mem_kib
        except BaseException:
            pass
        return None

    def __get_mem_size() -> Optional[int]:
        """
        Get the memory size information from any available source.

        :returns: an integer with the memory size if available
        :rtype: Optional[int]
        """
        s = __get_mem_size_sysconf()
        if s is None:
            return __get_mem_size_meminfo()
        return s

    # log all information in memory to convert it to one constant string.
    with InMemoryLogger() as imr:
        with imr.key_values(logging.SECTION_SYS_INFO) as kv:
            with kv.scope(logging.SCOPE_SESSION) as k:
                __v(k, logging.KEY_SESSION_START, datetime.now())
                __v(k, logging.KEY_NODE_NAME, platform.node())

            with kv.scope(logging.SCOPE_VERSIONS) as k:
                __v(k, "moptipy", ver.__version__)
                for package in ["numpy", "numba", "matplotlib",
                                "scikit-learn"]:
                    __v(k, package.replace("-", ""),
                        ilm.version(package).strip())

            with kv.scope(logging.SCOPE_HARDWARE) as k:
                __v(k, logging.KEY_HW_N_CPUS, os.cpu_count())
                __v(k, logging.KEY_HW_BYTE_ORDER, sys.byteorder)
                __v(k, logging.KEY_HW_MACHINE, platform.machine())
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
    return "\n".join(lst[1:(len(lst) - 1)])


#: The internal variable holding the memory information
__SYS_INFO: Final[str] = __make_sys_info()
del __make_sys_info  # delete the no-longer-needed function


def log_sys_info(logger: Logger) -> None:
    """
    Write the system information section to a logger.

    The concept of this method is that we only construct the whole system
    configuration exactly once in memory and then always directly flush it
    as a string to the logger. This is much more efficient than querying it
    every single time.

    :param Logger logger: the logger
    """
    with logger.text(logging.SECTION_SYS_INFO) as txt:
        txt.write(__SYS_INFO)
