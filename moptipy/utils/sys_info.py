"""A tool for writing a section with system information into log files."""
import importlib.metadata as ilm
import os
import platform
import re
import socket
import sys
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Final

import psutil  # type: ignore

import moptipy.version as ver
from moptipy.api import logging
from moptipy.utils.logger import InMemoryLogger, Logger, KeyValueLogSection
from moptipy.utils.path import Path


# noinspection PyBroadException
def __make_sys_info() -> str:
    """
    Build the system info string.

    This method is only used once and then deleted.

    :returns: the system info string.
    """
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

    # noinspection PyBroadException
    def __get_processor_name() -> Optional[str]:
        """
        Get the processor name.

        :returns: a string if there is any processor information
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
        """
        vs = __get_mem_size_sysconf()
        if vs is None:
            vs = __get_mem_size_meminfo()
        if vs is None:
            vs = psutil.virtual_memory().total
        return vs

    # log all information in memory to convert it to one constant string.
    with InMemoryLogger() as imr:
        with imr.key_values(logging.SECTION_SYS_INFO) as kv:
            with kv.scope(logging.SCOPE_SESSION) as k:
                __v(k, logging.KEY_SESSION_START, datetime.now())
                __v(k, logging.KEY_NODE_NAME, platform.node())
                proc = psutil.Process()
                __v(k, logging.KEY_PROCESS_ID, hex(proc.pid))
                cpua = proc.cpu_affinity()
                if cpua:
                    __v(k, logging.KEY_CPU_AFFINITY, ",".join(map(str, cpua)))
                del proc, cpua

                # see https://stackoverflow.com/questions/166506/.
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # doesn't even have to be reachable
                    s.connect(('10.255.255.255', 1))
                    ip = s.getsockname()[0]
                except BaseException:
                    ip = '127.0.0.1'
                finally:
                    s.close()
                __v(k, logging.KEY_NODE_IP, ip)

            with kv.scope(logging.SCOPE_VERSIONS) as k:
                __v(k, "moptipy", ver.__version__)
                for package in ["numpy", "numba", "matplotlib",
                                "psutil", "scikit-learn"]:
                    __v(k, package.replace("-", ""),
                        ilm.version(package).strip())

            with kv.scope(logging.SCOPE_HARDWARE) as k:
                __v(k, logging.KEY_HW_MACHINE, platform.machine())
                __v(k, logging.KEY_HW_N_PHYSICAL_CPUS,
                    psutil.cpu_count(logical=False))
                __v(k, logging.KEY_HW_N_LOGICAL_CPUS,
                    psutil.cpu_count(logical=True))

                # store the CPU speed information
                cpuf: Dict[Tuple[int, int], int] = {}
                total: int = 0
                for cf in psutil.cpu_freq(True):
                    t = (int(cf.min), int(cf.max))
                    cpuf[t] = cpuf.get(t, 0) + 1
                    total += 1
                memlst: List[Tuple[int, ...]]
                if total > 1:
                    memlst = [(key[0], key[1], value) for
                              key, value in cpuf.items()]
                    memlst.sort()
                else:
                    memlst = list(cpuf)

                def __make_mhz_str(tpl: Tuple[int, ...]) -> str:
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
    return "\n".join(lst[1:(len(lst) - 1)])


#: The internal variable holding the system information
__SYS_INFO: Final[List[str]] = [__make_sys_info()]


def refresh_sys_info():
    """Refresh the system information."""
    __SYS_INFO[0] = __make_sys_info()


def log_sys_info(logger: Logger) -> None:
    """
    Write the system information section to a logger.

    The concept of this method is that we only construct the whole system
    configuration exactly once in memory and then always directly flush it
    as a string to the logger. This is much more efficient than querying it
    every single time.

    :param logger: the logger
    """
    with logger.text(logging.SECTION_SYS_INFO) as txt:
        txt.write(__SYS_INFO[0])
