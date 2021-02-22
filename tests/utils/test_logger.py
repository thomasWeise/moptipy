from io import open
from os.path import isfile, exists

from moptipy.utils import TempFile, Logger


def test_log_files():
    with TempFile() as t:
        path = str(t)
        with Logger(path) as log:
            with log.csv("A", ["x", "y"]) as csv:
                csv.row([1, 2])
                csv.row([3.3, 12])
            with log.key_values("B") as kv:
                kv.key_value("a", 5)
                with kv.scope("b") as kv2:
                    kv2.key_value("c", "d")
                    with kv2.scope("e") as kv3:
                        kv3.key_value("f", "g")
                    kv2.key_value("h", "i")
                kv.key_value("j", "l")
            with log.text("C") as txt:
                txt.write("blabla")
            with log.csv("D", ["o", "p", "q"]) as csv:
                csv.row([-341, 42, 3])
                csv.row([4, 52, 12])
        result = open(path, "r").read().splitlines()
        assert result == ["BEGIN_A",
                          "x;y",
                          "1;2",
                          "3.3;12",
                          "END_A",
                          "BEGIN_B",
                          "a:5",
                          "b.c:d",
                          "b.e.f:g",
                          "b.h:i",
                          "j:l",
                          "END_B",
                          "BEGIN_C",
                          "blabla",
                          "END_C",
                          "BEGIN_D",
                          "o;p;q",
                          "-341;42;3",
                          "4;52;12",
                          "END_D"]
    assert not isfile(path)
    assert not exists(path)
