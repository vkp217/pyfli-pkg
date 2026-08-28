"""Tests for pyfli.io.SaveLoadDirector config-driven saver construction."""

import logging
import os

import pytest

from pyfli import DataSaver, SaveLoadDirector


@pytest.fixture
def data_dir(tmp_path):
    """A source-data directory the director can auto-derive a session from."""
    d = os.path.join(tmp_path, "mydata")
    os.makedirs(d)
    return d


# --------------------------------------------------------------------------
# config merging / validation
# --------------------------------------------------------------------------
def test_defaults_fill_missing_keys():
    d = SaveLoadDirector({"mode": None})
    assert d["MODEL_TYPE"] == "bi-exponential"
    assert d["bin_r"] == 0
    assert d["version"] == 1
    assert d["auto_path"] is True


def test_save_suffix_derived_from_bin_r_and_version():
    d = SaveLoadDirector({"mode": None, "bin_r": 3, "version": 7})
    assert d.save_suffix == "bin3_pf_Analysis_v7"
    assert d["save_suffix"] == "bin3_pf_Analysis_v7"


def test_kwarg_overrides_win_over_config():
    d = SaveLoadDirector({"mode": "save", "version": 1}, version=9, mode=None)
    assert d.mode is None
    assert d.save_suffix.endswith("_v9")


def test_passthrough_keys_are_preserved():
    d = SaveLoadDirector({"mode": None, "free_memory": False, "extra": 123})
    assert d["free_memory"] is False
    assert d["extra"] == 123
    assert d.to_dict()["extra"] == 123


@pytest.mark.parametrize("bad_mode", ["bogus", "SAVE", 0, ""])
def test_invalid_mode_raises(bad_mode):
    with pytest.raises(ValueError, match="mode"):
        SaveLoadDirector({"mode": bad_mode})


# --------------------------------------------------------------------------
# mode=None
# --------------------------------------------------------------------------
def test_mode_none_builds_no_saver(caplog):
    with caplog.at_level(logging.INFO, logger="pyfli"):
        saver = SaveLoadDirector({"mode": None}).build()
    assert saver is None
    assert "No data will be saved" in caplog.text


# --------------------------------------------------------------------------
# mode='save'
# --------------------------------------------------------------------------
def test_save_auto_path_creates_suffixed_sibling_folder(data_dir):
    d = SaveLoadDirector(
        {"mode": "save", "auto_path": True, "DATA_PATH": data_dir, "version": 2}
    )
    saver = d.build()
    assert isinstance(saver, DataSaver)
    assert saver.save_dir == data_dir + "_bin0_pf_Analysis_v2"
    assert os.path.isdir(saver.save_dir)


def test_save_auto_path_requires_data_path():
    d = SaveLoadDirector({"mode": "save", "auto_path": True, "DATA_PATH": None})
    with pytest.raises(ValueError, match="DATA_PATH"):
        d.build()


def test_save_manual_path_uses_root(tmp_path):
    root = os.path.join(tmp_path, "runs")
    os.makedirs(root)
    d = SaveLoadDirector(
        {
            "mode": "save",
            "auto_path": False,
            "manual_save_root_path": root,
            "version": 5,
        }
    )
    saver = d.build()
    assert saver.save_dir == DataSaver.resolve_path(root, "bin0_pf_Analysis_v5")
    assert os.path.isdir(saver.save_dir)


def test_save_manual_path_requires_root():
    d = SaveLoadDirector(
        {"mode": "save", "auto_path": False, "manual_save_root_path": None}
    )
    with pytest.raises(ValueError, match="manual_save_root_path"):
        d.build()


# --------------------------------------------------------------------------
# mode='load'
# --------------------------------------------------------------------------
def test_load_auto_path_reconnects_to_saved_session(data_dir):
    cfg = {"mode": "save", "auto_path": True, "DATA_PATH": data_dir, "version": 4}
    saver = SaveLoadDirector(cfg).build()
    saver.save_json("cfg", {"a": 1})

    loaded = SaveLoadDirector.from_config({**cfg, "mode": "load"})
    assert isinstance(loaded, DataSaver)
    assert loaded.save_dir == saver.save_dir
    assert loaded.load_json("cfg") == {"a": 1}


def test_load_manual_load_path_is_used_verbatim(tmp_path):
    existing = DataSaver(
        os.path.join(tmp_path, "sess"), folder_name="_pyfli_Analysis", new_session=True
    )
    loaded = SaveLoadDirector.from_config(
        {"mode": "load", "auto_path": False, "manual_load_path": existing.save_dir}
    )
    assert loaded.save_dir == existing.save_dir


def test_load_falls_back_to_save_root_when_no_load_path(tmp_path):
    root = os.path.join(tmp_path, "runs")
    save_cfg = {
        "mode": "save",
        "auto_path": False,
        "manual_save_root_path": root,
        "version": 6,
    }
    os.makedirs(root)
    saver = SaveLoadDirector(save_cfg).build()

    loaded = SaveLoadDirector.from_config({**save_cfg, "mode": "load"})
    assert loaded.save_dir == saver.save_dir


def test_load_manual_without_any_path_raises():
    d = SaveLoadDirector(
        {
            "mode": "load",
            "auto_path": False,
            "manual_load_path": None,
            "manual_save_root_path": None,
        }
    )
    with pytest.raises(ValueError, match="manual_load_path"):
        d.build()


def test_load_auto_path_requires_data_path():
    d = SaveLoadDirector({"mode": "load", "auto_path": True, "DATA_PATH": None})
    with pytest.raises(ValueError, match="DATA_PATH"):
        d.build()


# --------------------------------------------------------------------------
# misc surface
# --------------------------------------------------------------------------
def test_repr_mentions_mode_and_suffix():
    text = repr(SaveLoadDirector({"mode": "save", "version": 1}))
    assert "SaveLoadDirector" in text
    assert "mode='save'" in text
    assert "bin0_pf_Analysis_v1" in text


def test_get_returns_default_for_missing_key():
    d = SaveLoadDirector({"mode": None})
    assert d.get("nope", "fallback") == "fallback"
    assert d.get("mode", "fallback") is None


def test_file_data_path_used_as_is(tmp_path):
    f = os.path.join(tmp_path, "acq.sdt")
    with open(f, "w") as fh:
        fh.write("x")
    d = SaveLoadDirector({"mode": "save", "auto_path": True, "DATA_PATH": f})
    saver = d.build()
    # a file path is used as-is: its extension-stripped basename gets the
    # suffix appended directly (no trailing "_" like the directory branch adds)
    assert saver.save_dir == os.path.join(tmp_path, "acqbin0_pf_Analysis_v1")
