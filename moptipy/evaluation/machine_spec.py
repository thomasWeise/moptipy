"""Load machine specs from log files."""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Callable, Final, Iterable

from pycommons.io.csv import COMMENT_START
from pycommons.io.path import Path
from pycommons.types import type_error

from moptipy.api.logging import SECTION_SYS_INFO
from moptipy.utils.logger import SECTION_END, SECTION_START, InMemoryLogger
from moptipy.utils.sys_info import log_sys_info


@dataclass(frozen=True, init=True, order=True, eq=True)
class Machine:
    """
    An immutable record of machine information.

    >>> m = Machine(
    ...     machine_id="A",
    ...     ram_bytes=16853479424,
    ...     os="Ubuntu Linux",
    ...     cpu="Intel64 Family 6 Model 151",
    ...     cpu_mhz=2100,
    ...     python="3.12.13")
    >>> m.setup_str()
    'Python 3.12.13 on an Intel64 Family 6 Model 151 CPU at 2.1 GHz with \
15.7 GiB RAM and Ubuntu Linux'
    """

    #: The machine id
    machine_id: str

    #: the amount of memory
    ram_bytes: int | None = field(default=None)

    #: the operating system
    os: str | None = field(default=None)

    #: the CPU
    cpu: str | None = field(default=None)

    #: the mhz
    cpu_mhz: int | None = field(default=None)

    #: the python version
    python: str | None = field(default=None)

    def setup_str(self) -> str:
        """
        Get the system setup as string.

        :return: the setup string
        """
        result = ""
        if self.python is not None:
            result = f"Python {self.python}"

        if self.cpu is not None:
            if str.__len__(result) > 0:
                schr = "an" if str.lower(self.cpu[0]) in "aeiou" else "a"
                result = f"{result} on {schr} "
            result = f"{result}{self.cpu} CPU"
            if self.cpu_mhz is not None:
                speed = f"{self.cpu_mhz / 1000:.1f}".removesuffix(".0")
                result = f"{result} at {speed} GHz"
        if self.ram_bytes is not None:
            ram = f"{self.ram_bytes / 1073741824:.1f}".removesuffix(".0")
            if str.__len__(result) > 0:
                result = f"{result} with "
            result = f"{result}{ram} GiB RAM"
        if self.os is not None:
            if str.__len__(result) > 0:
                result = f"{result} and "
            result = f"{result}{self.os}"
        return result


#: the architecture key
__ARCH_KEY: Final[str] = "hardware.machine"
#: the machine key
__MACHINE_KEY: Final[str] = "session.node"
#: the CPU key
__CPU_KEY: Final[str] = "hardware.cpu"
#: the MHz key
__MHZ_KEY: Final[str] = "hardware.cpuMhz"
#: the ram key
__RAM_KEY: Final[str] = "hardware.memSize"
#: the os name
__PYTHON_KEY: Final[str] = "python.version"
#: the os name key
__OS_NAME_KEY: Final[str] = "os.name"
#: the os release key
__OS_RELEASE_KEY: Final[str] = "os.release"
#: the os version key
__OS_VERSION_KEY: Final[str] = "os.version"

#: the keys
__KEYS: Final[set[str]] = {
    __ARCH_KEY, __MACHINE_KEY, __CPU_KEY, __MHZ_KEY, __RAM_KEY, __PYTHON_KEY,
    __OS_NAME_KEY, __OS_RELEASE_KEY, __OS_VERSION_KEY}

#: the section start
__SEC_START: Final[str] = f"{SECTION_START}{SECTION_SYS_INFO}"
#: the section end
__SEC_END: Final[str] = f"{SECTION_END}{SECTION_SYS_INFO}"

#: drop this text
__CPU_DROP: Final[tuple[str, ...]] = (
    "genuineintel", "stepping", "authenticamd")
#: drop this text
__CPU_DROP_CORES: Final[tuple[str, ...]] = ("-core processor", )


def __load_machine(file: Path, result: dict[str, Machine]) -> None:
    """
    Load a machine record from a log file.

    :param file: the path to load from
    """
    state: int = 0
    data: dict[str, str] = {}
    with file.open_for_read() as stream:
        for srow in stream:
            row = str.strip(srow)
            if (str.__len__(row) <= 0) or row.startswith(COMMENT_START):
                continue
            if str.__eq__(row, __SEC_START):
                if state != 0:
                    raise ValueError(f"{__SEC_START!r} appears twice?")
                state = 1
                continue
            if str.__eq__(row, __SEC_END):
                if state != 1:
                    raise ValueError(f"{__SEC_END!r} before {__SEC_START!r}?")
                state = 2
                continue

            if state == 1:
                dot = str.find(row, ":")
                if dot > 0:
                    key = str.strip(row[:dot])
                    if key in __KEYS:
                        val = str.strip(row[dot + 1:])
                        if str.__len__(val) > 0:
                            data[key] = val

    if __MACHINE_KEY not in data:
        return
    machine_id = data[__MACHINE_KEY]
    if machine_id in result:
        return

    try:
        ram = int(data[__RAM_KEY])
    except (KeyError, ValueError):
        ram = None

    cpu: str | None = data.get(__CPU_KEY)
    if cpu is None:
        cpu = data.get(__ARCH_KEY)
    if cpu is not None:
        ncpu = cpu
        new_len = str.__len__(ncpu)
        old_len = new_len + 1
        while old_len > new_len:
            old_len = new_len
            ncpu_l = str.lower(ncpu)
            for d in __CPU_DROP:
                di = str.rfind(ncpu_l, d)
                if di > 0:
                    ncpu = str.strip(ncpu[:di])
                    ncpu_l = str.lower(ncpu)

            for d in __CPU_DROP_CORES:
                if ncpu_l.endswith(d):
                    ncpux = str.strip(ncpu[:-str.__len__(d)])
                    v = str.rfind(ncpux, " ")
                    if 0 < v < (str.__len__(ncpux) - 1):
                        ncpu = str.strip(ncpu[:v])

            ncpu = str.removesuffix(ncpu, ",")
            new_len = str.__len__(ncpu)
        if str.__len__(ncpu) > 0:
            cpu = ncpu

    mhz: int | None = None
    mhz_str: str | None = data.get(__MHZ_KEY)
    if mhz_str is not None:
        di = str.rfind(mhz_str, "*")
        if 0 < di < (str.__len__(mhz_str) - 1):
            mhz_str = str.strip(mhz_str[:di])
        mhz_str = str.strip(str.removesuffix(str.removeprefix(
            mhz_str, "("), ")"))
        di = str.rfind(mhz_str, ".")
        if 0 < di < (str.__len__(mhz_str) - 1):
            mhz_str = str.strip(mhz_str[di + 1:])
        mhz_str = mhz_str.removesuffix("MHz")
        with suppress(ValueError):
            mhz = int(mhz_str)

    python: str | None = data.get(__PYTHON_KEY)
    if python is not None:
        di = str.find(python, "|")
        if di > 0:
            python = str.strip(python[:di])
        di = str.find(python, " ")
        if di > 0:
            python = str.strip(python[:di])
        di = str.find(python, ".")
        if di > 0:
            si = str.find(python, ".", di + 1)
            if si > di:
                python = str.strip(python[:si])

    os: str | None = data.get(__OS_NAME_KEY)
    if os is not None:
        os_release: str | None = data.get(__OS_RELEASE_KEY)
        os_version: str | None = data.get(__OS_VERSION_KEY)

        if str.lower(os) == "linux":
            if os_version is not None:
                di = os_version.find(" ")
                if di > 0:
                    os_version = str.strip(os_version[:di])
                os = f"{os_version} Linux"
            if os_release is not None:
                di = os_release.find("-")
                if di > 0:
                    os_release = str.strip(os_release[:di])
                os = f"{os}, {os_release} Kernel"
        elif os_release is not None:
            os = f"{os} {os_release}"
        elif os_version is not None:
            os = f"{os} {os_version}"

    result[machine_id] = Machine(
        machine_id=machine_id,
        ram_bytes=ram,
        os=os,
        cpu=cpu,
        cpu_mhz=mhz,
        python=python)


def load_machines(source: str,
                  result: dict[str, Machine]) -> None:
    """
    Load the machine data from logs and store them under the given dictionary.

    :param source: the path to load from
    :param result: the dictionary of machine-IDs and machine records.
    """
    path: Final[Path] = Path(source)
    if not isinstance(result, dict):
        raise type_error(result, "result", dict)
    if path.is_dir():
        for sub in path.list_dir():
            load_machines(sub, result)
    elif path.is_file():
        with path.open_for_read() as stream:
            machine = get_machine(stream, result.__contains__)
            if machine is not None:
                result[machine.machine_id] = machine
    else:
        raise ValueError(
            f"{path!r} identifies neither a file nor a directory.")


def get_machine(
        stream: Iterable[str],
        can_skip: Callable[[str], bool] = lambda _: False) -> Machine | None:
    """
    Load a machine data record from a stream of strings.

    This function parses a `stream` of strings, which could be from a log
    file, and extracts the core data of a machine.
    It then returns this data as a machine record.

    Optionally, a function `can_skip` can be provided.
    Each machine record has as ID usually the session node, i.e., the name of
    the corresponding computer.
    The function `can_skip` receives this as parameter and is called once.
    It can then decide whether the data should be fully parsed and a record
    should be returned (by returning `False`) or whether parsing can be
    aborted and `None` shall be returned.

    :param stream: the stream of strings
    :param can_skip: a function receiving a machine ID and returning `True` if
        `None` should be returned for this machine ID, i.e., if the record does
        not need to be parsed, and `False` otherwise
    """
    if not callable(can_skip):
        raise type_error(can_skip, "can_skip", call=True)
    if not isinstance(stream, Iterable):
        raise type_error(stream, "stream", Iterable)

    state: int = 0
    data: dict[str, str] = {}
    for srow in stream:
        row = str.strip(srow)
        if (str.__len__(row) <= 0) or row.startswith(COMMENT_START):
            continue
        if str.__eq__(row, __SEC_START):
            if state != 0:
                raise ValueError(f"{__SEC_START!r} appears twice?")
            state = 1
            continue
        if str.__eq__(row, __SEC_END):
            if state != 1:
                raise ValueError(f"{__SEC_END!r} before {__SEC_START!r}?")
            state = 2
            continue

        if state == 1:
            dot = str.find(row, ":")
            if dot > 0:
                key = str.strip(row[:dot])
                if key in __KEYS:
                    val = str.strip(row[dot + 1:])
                    if (key == __MACHINE_KEY) and can_skip(__MACHINE_KEY):
                        return None
                    if str.__len__(val) > 0:
                        data[key] = val

    machine_id = data.get(__MACHINE_KEY)
    if machine_id is None:
        return None

    try:
        ram = int(data[__RAM_KEY])
    except (KeyError, ValueError):
        ram = None

    cpu: str | None = data.get(__CPU_KEY)
    if cpu is None:
        cpu = data.get(__ARCH_KEY)
    if cpu is not None:
        ncpu = cpu
        new_len = str.__len__(ncpu)
        old_len = new_len + 1
        while old_len > new_len:
            old_len = new_len
            ncpu_l = str.lower(ncpu)
            for d in __CPU_DROP:
                di = str.rfind(ncpu_l, d)
                if di > 0:
                    ncpu = str.strip(ncpu[:di])
                    ncpu_l = str.lower(ncpu)

            for d in __CPU_DROP_CORES:
                if ncpu_l.endswith(d):
                    ncpux = str.strip(ncpu[:-str.__len__(d)])
                    v = str.rfind(ncpux, " ")
                    if 0 < v < (str.__len__(ncpux) - 1):
                        ncpu = str.strip(ncpu[:v])

            ncpu = str.removesuffix(ncpu, ",")
            new_len = str.__len__(ncpu)
        if str.__len__(ncpu) > 0:
            cpu = ncpu

    mhz: int | None = None
    mhz_str: str | None = data.get(__MHZ_KEY)
    if mhz_str is not None:
        di = str.rfind(mhz_str, "*")
        if 0 < di < (str.__len__(mhz_str) - 1):
            mhz_str = str.strip(mhz_str[:di])
        mhz_str = str.strip(str.removesuffix(str.removeprefix(
            mhz_str, "("), ")"))
        di = str.rfind(mhz_str, ".")
        if 0 < di < (str.__len__(mhz_str) - 1):
            mhz_str = str.strip(mhz_str[di + 1:])
        mhz_str = mhz_str.removesuffix("MHz")
        with suppress(ValueError):
            mhz = int(mhz_str)

    python: str | None = data.get(__PYTHON_KEY)
    if python is not None:
        di = str.find(python, "|")
        if di > 0:
            python = str.strip(python[:di])
        di = str.find(python, " ")
        if di > 0:
            python = str.strip(python[:di])
        di = str.find(python, ".")
        if di > 0:
            si = str.find(python, ".", di + 1)
            if si > di:
                python = str.strip(python[:si])

    os: str | None = data.get(__OS_NAME_KEY)
    if os is not None:
        os_release: str | None = data.get(__OS_RELEASE_KEY)
        os_version: str | None = data.get(__OS_VERSION_KEY)

        if str.lower(os) == "linux":
            if os_version is not None:
                di = os_version.find(" ")
                if di > 0:
                    os_version = str.strip(os_version[:di])
                os = f"{os_version} Linux"
            if os_release is not None:
                di = os_release.find("-")
                if di > 0:
                    os_release = str.strip(os_release[:di])
                os = f"{os}, {os_release} Kernel"
        elif os_release is not None:
            os = f"{os} {os_release}"
        elif os_version is not None:
            os = f"{os} {os_version}"

    return Machine(
        machine_id=machine_id,
        ram_bytes=ram,
        os=os,
        cpu=cpu,
        cpu_mhz=mhz,
        python=python)


def current_machine_spec() -> Machine:
    """
    Get the current machine specification.

    :returns: the specification of the current machine

    >>> a = current_machine_spec()
    >>> a is not None
    True
    >>> a is current_machine_spec()
    True
    """
    the_object = current_machine_spec
    the_property = "_machine_spec"
    if hasattr(the_object, the_property):
        return getattr(the_object, the_property)

    with InMemoryLogger() as ml:
        log_sys_info(ml)
        result = get_machine(ml.get_log())

    if result is None:
        raise ValueError("Did not get machine?")

    setattr(the_object, the_property, result)
    return result
