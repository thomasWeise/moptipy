from moptipy.utils import TempFile, TempDir, file_create_or_fail, \
    file_create_or_truncate, canonicalize_path, file_ensure_exists
from os.path import isfile, isdir, exists, dirname, sep, basename, getsize
from io import open
# noinspection PyPackageRequirements
from pytest import raises


def test_temp_file():
    with TempFile() as t:
        path = str(t)
        assert isinstance(path, str)
        assert len(path) > 0
        assert isfile(path)
        assert exists(path)
    assert not isfile(path)
    assert not exists(path)

    with TempFile(prefix="aaaa", suffix=".xxx") as t:
        path = str(t)
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
    with TempDir() as d:
        path = str(d)
        assert isinstance(path, str)
        assert len(path) > 0
        assert isdir(path)
        assert exists(path)
    assert not isdir(path)
    assert not exists(path)

    with TempDir() as d:
        path = str(d)
        assert isinstance(path, str)
        assert len(path) > 0
        assert isdir(path)
        assert exists(path)
        with TempFile(d) as f:
            path2 = str(f)
            assert isinstance(path2, str)
            assert dirname(path2) == path
            assert len(path2) > 0
            assert isfile(path2)
            assert exists(path2)
        with TempFile(path) as f:
            path2 = str(f)
            assert isinstance(path2, str)
            assert dirname(path2) == path
            assert len(path2) > 0
            assert isfile(path2)
            assert exists(path2)
        inner = (path + sep + "xx.y")
        open(inner, "w").close()
        assert isfile(inner)
        assert exists(inner)
    assert not isdir(path)
    assert not exists(path)
    assert not exists(path2)
    assert not exists(inner)


def test_file_create_or_fail():
    with TempDir() as td:
        tds = str(td)
        s = canonicalize_path(tds + sep + "a.txt")
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        s = file_create_or_fail(s)
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0
        with raises(ValueError):
            file_create_or_fail(s)


def test_file_create_or_truncate():
    with TempDir() as td:
        tds = str(td)
        s = canonicalize_path(tds + sep + "a.txt")
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        s = file_create_or_truncate(s)
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0

        with open(s, "w") as f:
            f.write("blablabla")

        assert getsize(s) > 0

        s2 = file_create_or_truncate(s)
        assert s == s2
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0


def test_file_ensure_exists():
    with TempDir() as td:
        tds = str(td)
        s = canonicalize_path(tds + sep + "a.txt")
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert not exists(s)
        s, existed = file_ensure_exists(s)
        assert isinstance(s, str)
        assert len(s) > 0
        assert s.startswith(tds)
        assert s.endswith("a.txt")
        assert exists(s)
        assert isfile(s)
        assert getsize(s) == 0
        assert not existed

        with open(s, "w") as f:
            f.write("blablabla")

        old_size = getsize(s)
        assert old_size > 0

        s2, existed = file_ensure_exists(s)
        assert isinstance(s2, str)
        assert s2 == s
        assert exists(s2)
        assert isfile(s2)
        assert getsize(s2) == getsize(s)
        assert existed
