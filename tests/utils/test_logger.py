"""Test the structured logger."""
import csv
from os.path import exists, isfile

# noinspection PyPackageRequirements
import yaml

from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


def test_log_files() -> None:
    """Test the generation of log files."""
    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.csv("A", ["x", "y"]) as xsv:
                xsv.row([1, 2])
                xsv.row([3.3, 12])
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
            with log.csv("D", ["o", "p", "q"]) as xsv:
                xsv.row([-341, 42, 3])
                xsv.row([4, 52, 12])
        with open(path) as file:
            result = file.read().splitlines()
        assert result == ["BEGIN_A",  # 0
                          "x;y",  # 1
                          "1;2",  # 2
                          "3.3;12",  # 3
                          "END_A",  # 4
                          "BEGIN_B",  # 5
                          "a: 5",  # 6
                          "b.c: d",  # 7
                          "b.e.f: g",  # 8
                          "b.h: i",  # 9
                          "j: l",  # 10
                          "END_B",  # 11
                          "BEGIN_C",  # 12
                          "blabla",  # 13
                          "END_C",  # 14
                          "BEGIN_D",  # 15
                          "o;p;q",  # 16
                          "-341;42;3",  # 17
                          "4;52;12",  # 18
                          "END_D"]  # 19
    kv_part = result[6:11]
    dic = yaml.safe_load("\n".join(kv_part))
    assert dic["a"] == 5
    assert dic["b.c"] == "d"
    assert dic["b.e.f"] == "g"
    assert dic["b.h"] == "i"
    assert dic["j"] == "l"
    assert len(dic) == 5

    csv_part_1 = result[1:4]
    rows = []
    for r in csv.reader(csv_part_1, delimiter=";"):
        rows.append(r)
    assert rows == [["x", "y"],
                    ["1", "2"],
                    ["3.3", "12"]]

    csv_part_2 = result[16:19]
    rows = []
    for r in csv.reader(csv_part_2, delimiter=";"):
        rows.append(r)
    assert rows == [["o", "p", "q"],
                    ["-341", "42", "3"],
                    ["4", "52", "12"]]

    assert not isfile(path)
    assert not exists(path)
