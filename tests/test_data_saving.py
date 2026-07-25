"""Tests for pyfli.io.DataSaver save/load/reconnect symmetry."""

import os

import numpy as np
import pytest

from pyfli import DataSaver


@pytest.fixture
def base_path(tmp_path):
    return os.path.join(tmp_path, "exp1")


def test_new_session_creates_suffixed_folder(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis", new_session=True)
    assert saver.save_dir == base_path + "_pyfli_Analysis"
    assert os.path.isdir(saver.save_dir)
    assert os.path.isfile(saver.log_file)


def test_resolve_path_matches_actual_save_dir(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis", new_session=False)
    resolved = DataSaver.resolve_path(base_path, "_pyfli_Analysis")
    assert resolved == saver.save_dir


def test_log_to_file_writes_without_console_echo(base_path, capsys, caplog):
    import logging

    saver = DataSaver(base_path, folder_name="_pyfli_Analysis")
    capsys.readouterr()  # discard anything from __init__

    with caplog.at_level(logging.INFO, logger="pyfli"):
        saver.log_to_file("quiet table row")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert caplog.records == []

    with open(saver.log_file) as f:
        assert "quiet table row" in f.read()


def test_json_save_and_load_roundtrip(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis")
    saver.save_json("cfg", {"a": 1, "b": "x"})
    loaded = saver.load_json("cfg")
    assert loaded == {"a": 1, "b": "x"}


def test_npy_array_save_and_load_roundtrip(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis")
    arr = np.array([1, 2, 3])
    saver.save_npy("arr", arr)
    loaded = saver.load_npy("arr")
    np.testing.assert_array_equal(loaded, arr)


def test_npy_dict_save_and_load_roundtrip(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis")
    d = {"tau1": np.array([1.0, 2.0]), "note": "hi"}
    saver.save_npy("d", d)
    loaded = saver.load_npy("d")
    assert isinstance(loaded, dict)
    assert loaded["note"] == "hi"
    np.testing.assert_array_equal(loaded["tau1"], d["tau1"])


def test_config_save_and_load_roundtrip(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis")
    saver.save_config({"model_type": "bi-exponential", "n_jobs": 4}, name="cfg2")
    loaded = saver.load_config(name="cfg2")
    assert loaded == {"model_type": "bi-exponential", "n_jobs": 4}


def test_load_classmethod_reconnects_to_existing_session(base_path):
    saver = DataSaver(base_path, folder_name="_pyfli_Analysis", new_session=True)
    saver.save_json("cfg", {"a": 1})

    reconnected = DataSaver.load(saver.save_dir)
    assert reconnected.save_dir == saver.save_dir
    assert reconnected.load_json("cfg") == {"a": 1}


def test_folder_name_none_requires_existing_directory(tmp_path):
    missing = os.path.join(tmp_path, "does_not_exist")
    with pytest.raises(FileNotFoundError):
        DataSaver(missing, folder_name=None)
