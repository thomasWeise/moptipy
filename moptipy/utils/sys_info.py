"""A tool for writing a section with system information into log files."""
import os
import platform
import re
import sys
from datetime import datetime
from typing import Optional

import numpy

import moptipy.version
from moptipy.utils import logging
from moptipy.utils.logger import InMemoryLogger, Logger, KeyValueSection
from moptipy.utils.path import Path


def __make_sys_info() -> str:
    def __v(sec: KeyValueSection, key: str, value):
        if value is None:
            return
        value = " ".join([s.strip() for s in
                          str(value).strip().split("\n")]).strip()
        if len(value) <= 0:
            return
        sec.key_value(key, value)

    # noinspection PyBroadException
    def __get_processor_name() -> Optional[str]:
        try:
            if platform.system() == "Windows":
                return platform.processor()
            if platform.system() == "Linux":
                for line in Path.path("/proc/cpuinfo").read_all_list():
                    if "model name" in line:
                        return re.sub(".*model name.*:", "",
                                      line, 1).strip()
        except BaseException:
            pass
        return None

    # noinspection PyBroadException
    def __get_mem_size_sysconf() -> Optional[int]:
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
    def __get_mem_size_memconf() -> Optional[int]:
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
        s = __get_mem_size_sysconf()
        if s is None:
            return __get_mem_size_memconf()
        return s

    with InMemoryLogger() as imr:
        with imr.key_values(logging.SECTION_SYS_INFO) as kv:
            with kv.scope(logging.SCOPE_SESSION) as k:
                __v(k, logging.KEY_SESSION_START, datetime.now())

            with kv.scope(logging.SCOPE_VERSIONS) as k:
                __v(k, logging.KEY_MOPTIPY_VERSION,
                    moptipy.version.__version__)
                __v(k, logging.KEY_NUMPY_VERSION, numpy.__version__)

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
                __v(k, logging.KEY_OS_VERSION, platform.release())

        lst = imr.get_log()

    if len(lst) < 3:
        raise ValueError("sys info turned out to be empty?")
    return "\n".join(lst[1:(len(lst) - 1)])


__SYS_INFO = __make_sys_info()
del __make_sys_info


def log_sys_info(logger: Logger) -> None:
    """
    Write the system information section to a logger.

    :param logger: the logger
    """
    with logger.text(logging.SECTION_SYS_INFO) as txt:
        txt.write(__SYS_INFO)
