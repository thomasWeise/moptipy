import os
import platform
import re
import subprocess
import sys
from datetime import datetime

import numpy

import moptipy.version
from moptipy.utils import logging
from moptipy.utils.logger import InMemoryLogger, Logger, KeyValueSection


def __make_sys_info():
    def __v(sec: KeyValueSection, key: str, value):
        if value is None:
            return
        value = " ".join([s.strip() for s in
                          str(value).strip().split("\n")]).strip()
        if len(value) <= 0:
            return
        sec.key_value(key, value)

    def __get_processor_name():
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).decode().strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True)\
                .decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()
        return ""

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


def log_sys_info(logger: Logger):
    """
    The method for writing the system information section to a logger
    :param logger: the logger
    """
    with logger.text(logging.SECTION_SYS_INFO) as txt:
        txt.write(__SYS_INFO)
