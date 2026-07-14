import numpy as np
import tifffile

from neuro_storm.cli.load_raw import find_tiffs, load_raw


def test_load_raw_directory(tmp_path):
    a = np.zeros((4, 4), dtype=np.uint16)
    b = np.ones((2, 8, 8), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "plane_00.tif", a)
    tifffile.imwrite(tmp_path / "plane_01.tif", b)

    arrays = load_raw(tmp_path)

    assert len(arrays) == 2
    assert arrays[0].shape == (4, 4)
    assert arrays[1].shape == (2, 8, 8)


def test_load_raw_single_file(tmp_path):
    path = tmp_path / "single.tiff"
    tifffile.imwrite(path, np.zeros((3, 3), dtype=np.uint8))

    arrays = load_raw(path)

    assert len(arrays) == 1
    assert arrays[0].shape == (3, 3)


def test_find_tiffs_sorted(tmp_path):
    for name in ("b.tif", "a.tif", "c.tiff"):
        tifffile.imwrite(tmp_path / name, np.zeros((1, 1), dtype=np.uint8))

    names = [p.name for p in find_tiffs(tmp_path)]

    assert names == ["a.tif", "b.tif", "c.tiff"]
