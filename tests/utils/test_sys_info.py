"""Test the sys-info and logging."""

from typing import Final

from moptipy.utils.logger import InMemoryLogger
from moptipy.utils.sys_info import log_sys_info, refresh_sys_info


def __check_sys_info() -> None:
    """Test the system information."""
    with InMemoryLogger() as logger:
        log_sys_info(logger)
        log: list[str] = logger.get_log()

    assert log[0] == "BEGIN_SYS_INFO"
    assert log[-1] == "END_SYS_INFO"
    assert len(log) > 2
    log = log[1:-1]

    values: Final[dict[str, str]] = {}
    sep: Final[str] = ": "
    for line in log:
        i = line.find(sep)
        assert 0 < i < (len(line) - 1)
        key = line[:i]
        assert key == key.strip()
        assert key not in values
        assert len(key) > 0
        value = line[i + len(sep):]
        assert value == value.strip()
        assert len(value) > 0
        values[key] = value

    assert "session.start" in values
    assert "session.node" in values
    assert "session.cpuAffinity" in values
    assert "session.procesId" in values
    assert "session.ipAddress" in values
    assert "version.cycler" in values
    assert "version.fonttools" in values
    assert "version.joblib" in values
    assert "version.kiwisolver" in values
    assert "version.llvmlite" in values
    assert "version.matplotlib" in values
    assert "version.numba" in values
    assert "version.numpy" in values
    assert "version.packaging" in values
    assert "version.Pillow" in values
    assert "version.psutil" in values
    assert "version.pyparsing" in values
    assert "version.pythondateutil" in values
    assert "version.scikitlearn" in values
    assert "version.scipy" in values
    assert "version.six" in values
    assert "hardware.nPhysicalCpus" in values
    assert "hardware.nLogicalCpus" in values
    assert "hardware.cpuMhz" in values
    assert "hardware.byteOrder" in values
    assert "hardware.cpu" in values
    assert "hardware.memSize" in values
    assert "python.version" in values
    assert "python.implementation" in values
    assert "os.name" in values
    assert "os.release" in values
    assert "os.version" in values


def test_and_renew_sys_info() -> None:
    """Test the sys info after renewing it."""
    __check_sys_info()
    refresh_sys_info()
    __check_sys_info()
