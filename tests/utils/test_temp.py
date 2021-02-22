from moptipy.utils import TempFile, TempDir
from os.path import isfile, isdir, exists, dirname, sep, basename
from io import open


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
