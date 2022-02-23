"""Test the interaction with the file system and temp files."""
from io import open
from os.path import isfile, isdir, exists, dirname, basename, getsize, join

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.utils.path import Path
from moptipy.utils.temp import TempFile, TempDir


def test_temp_file():
    """Test the creation and deletion of temporary files."""
    with TempFile.create() as path:
        assert isinstance(path, str)
        assert len(path) > 0
        assert isfile(path)
        assert exists(path)
    assert not isfile(path)
    assert not exists(path)

    with TempFile.create(prefix="aaaa", suffix=".xxx") as path:
        assert isinstance(path, str)
        assert len(path) > 0
        assert isfile(path)
        assert exists(path)
        bn = basename(path)
        assert bn.startswith("aaaa")
        assert bn.endswith(".xxx")
    assert not isfile(path)
    assert not exists(path)


def test_temp_dir():
    """Test the creation and deletion of temporary directories."""
    with TempDir.create() as path:
        assert isinstance(path, str)
        assert len(path) > 0
        assert isdir(path)
        assert exists(path)
    assert not isdir(path)
    assert not exists(path)

    with TempDir.create() as path:
        assert isinstance(path, str)
        assert len(path) > 0
        assert isdir(path)
        assert exists(path)
        with TempFile.create(path) as path2:
            assert isinstance(path2, str)
            assert dirname(path2) == path
            assert len(path2) > 0
            assert isfile(path2)
            assert exists(path2)
        with TempFile.create(path) as path2:
            assert isinstance(path2, str)
            assert dirname(path2) == path
            assert len(path2) > 0
            assert isfile(path2)
            assert exists(path2)
        inner = join(path, "xx.y")
        open(inner, "w").close()
        assert isfile(inner)
        assert exists(inner)
    assert not isdir(path)
    assert not exists(path)
    assert not exists(path2)
    assert not exists(inner)


def test_file_create_or_fail():
    with TempDir.create() as tds:
        s = Path.path(join(tds, "a.txt"))
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        s.create_file_or_fail()
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0
        with raises(ValueError):
            s.create_file_or_fail()


def test_file_create_or_truncate():
    with TempDir.create() as tds:
        s = Path.path(join(tds, "a.txt"))
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        s.create_file_or_truncate()
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0

        s.write_all("blablabla")

        assert getsize(s) > 0

        s.create_file_or_truncate()
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0


def test_file_ensure_exists():
    with TempDir.create() as tds:
        s = Path.path(join(tds, "a.txt"))
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        existed = s.ensure_file_exists()
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0
        assert not existed

        s.write_all("blablabla")

        old_size = getsize(s)
        assert old_size > 0

        existed = s.ensure_file_exists()
        assert exists(s)
        assert isfile(s)
        assert existed
