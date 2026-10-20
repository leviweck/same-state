"""config.example.toml lists every setting in config/default.yaml, loads, and holds dummy values only."""
import pathlib
import re

import yaml

from plm_kit import cli, config

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "config.example.toml"


def keys(d, pre=""):
    out = set()
    for k, v in d.items():
        out.add(pre + k)
        if isinstance(v, dict):
            out |= keys(v, f"{pre}{k}.")
    return out


def test_example_has_every_setting_and_loads():
    ex = config.load_config(EX)
    default = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    text = EX.read_text()
    missing = keys(default) - keys(ex) - {"retrieval.dense.custom"}
    assert missing == set(), missing
    assert "custom = {" in text                  # documented (commented) since TOML can't express null
    assert keys(ex) - keys(default) == set()
    assert ex["retrieval"]["mode"] == "fused" and ex["retrieval"]["not_in_notes_threshold"] == 0.78
    assert ex["data"]["index_dir"] is None and ex["sources"]["onedrive"]["client_id"] is None


def test_example_values_are_dummies():
    text = EX.read_text()
    assert "/path/to/your" in text
    assert not re.search(r"(?i)/(home|Users)/[a-z]", text)
    assert not re.search(r"(?i)(token|secret|password|key)\s*=\s*\"[^\"]+\"", text.replace('token_cache = ""', ""))


def test_cli_accepts_toml_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    f = tmp_path / "my.toml"
    f.write_text(EX.read_text().replace('docs_dir = "/path/to/your/notes"', 'docs_dir = "sample_data/docs"'))
    assert cli.main(["--config", str(f), "ask", "factory reset AX-7731", "--k", "1"]) == 0
    assert "home-wifi.md" in capsys.readouterr().out
