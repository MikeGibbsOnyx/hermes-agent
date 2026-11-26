import importlib.metadata
import argparse
import json
import logging
import os
from types import SimpleNamespace

import pytest

from hermes_cli import plugins_cmd


def _args(**kwargs):
    defaults = {
        "enabled": False,
        "user": False,
        "no_bundled": False,
        "plain": False,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_filter_plugin_entries_enabled_only():
    entries = [
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
        ("old-plugin", "1.0.0", "Old", "user", None, "old-plugin"),
    ]

    filtered = plugins_cmd._filter_plugin_entries(
        entries,
        _args(enabled=True),
        enabled={"disk-cleanup", "web-search-plus"},
        disabled={"old-plugin"},
    )

    assert [entry[0] for entry in filtered] == ["disk-cleanup", "web-search-plus"]


def test_cmd_list_plain_compact_output(monkeypatch, capsys):
    entries = [
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"web-search-plus"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_list(_args(plain=True, no_bundled=True))

    out = capsys.readouterr().out
    assert "web-search-plus" in out
    assert "enabled" in out
    assert "disk-cleanup" not in out
    assert "Search" not in out  # plain mode stays compact, no descriptions


def test_discover_all_plugins_includes_entrypoint_plugins(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()

    dist = SimpleNamespace(
        version="0.1.0",
        metadata={"Summary": "Karpathy-style LLM Wikis for Hermes"},
    )
    entry_point = SimpleNamespace(
        name="wiki",
        value="adapters.hermes.cli_plugin",
        group="hermes_agent.plugins",
        dist=dist,
    )

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir",
        lambda: bundled_dir,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: [entry_point],
    )

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "wiki",
            "0.1.0",
            "Karpathy-style LLM Wikis for Hermes",
            "entrypoint",
            "adapters.hermes.cli_plugin",
            "wiki",
        )
    ]


def test_declared_capabilities_for_entrypoint_uses_distribution_metadata(
    monkeypatch, tmp_path
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()
    plugin_ep = SimpleNamespace(
        name="thread-namer",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugins",
        dist=SimpleNamespace(version="1.0", metadata={"Summary": ""}),
    )
    capability_ep = SimpleNamespace(
        name="thread-namer.gateway.platform_actions",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugin_capabilities",
    )
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: [plugin_ep, capability_ep],
    )

    assert plugins_cmd._declared_capabilities_for_key("thread-namer") == [
        "gateway.platform_actions"
    ]


def test_entrypoint_spec_does_not_abort_plugin_status(monkeypatch):
    """An entry-point row stores ``module:attr`` in the path slot. Windows rejects that name
    (WinError 123); status must not open it as a plugin directory and abort the whole list."""
    from pathlib import Path

    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if str(self).startswith("mnemosyne_hermes:register"):
            err = OSError(22, "The filename, directory name, or volume label syntax is incorrect")
            err.winerror = 123
            raise err
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)

    assert plugins_cmd._plugin_status(
        "mnemosyne_hermes", set(), set(), key="mnemosyne_hermes",
        source="entrypoint", dir_path="mnemosyne_hermes:register",
    ) == "not enabled"


@pytest.mark.platforms("posix")  # chmod is a no-op on Windows
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores file permissions")
def test_unreadable_plugin_dir_is_skipped_by_every_manifest_scan(monkeypatch, tmp_path, caplog):
    """One plugin directory the process cannot stat() into (Windows WinError 5, POSIX mode 000)
    must be warned about and skipped — not abort discovery for every other plugin (#111804).
    Covers the loader scan (``scan_directory``) and the list/hub scan (``_scan_level``)."""
    from hermes_cli.plugins_discovery import scan_directory

    user_dir = tmp_path / "plugins"
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    for name in ("denied", "good"):
        (user_dir / name).mkdir(parents=True)
        (user_dir / name / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
    (user_dir / "denied").chmod(0)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr(importlib.metadata, "entry_points", list)

    try:
        with caplog.at_level(logging.WARNING):
            loader_names = [m.name for m in scan_directory(user_dir, "user")]
            listed_names = [entry[0] for entry in plugins_cmd._discover_all_plugins()]
    finally:
        (user_dir / "denied").chmod(0o700)

    assert loader_names == ["good"]
    assert listed_names == ["good"]
    assert caplog.text.count("Skipping unreadable plugin directory") == 2


@pytest.mark.skipif(plugins_cmd._resolve_git_executable() is None, reason="needs git")
def test_list_json_reports_catalog_drift_for_user_installs(monkeypatch, tmp_path, capsys):
    """A ``source: user`` install whose name matches a bundled catalog entry gets an additive
    ``update_available``: git HEAD vs the pinned sha, else plugin.yaml version vs catalog version."""
    import subprocess

    user_dir, bundled_dir, catalog_dir = tmp_path / "plugins", tmp_path / "bundled", tmp_path / "catalog"
    bundled_dir.mkdir()
    catalog_dir.mkdir()
    for name, version in (("copy-same", "0.1.3"), ("copy-old", "0.1.0"), ("git-pinned", "1.0.0"),
                          ("git-off-pin", "1.0.0"), ("not-in-catalog", "1.0.0")):
        (user_dir / name).mkdir(parents=True)
        (user_dir / name / "plugin.yaml").write_text(f"name: {name}\nversion: {version}\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    heads = {}
    for name in ("git-pinned", "git-off-pin"):
        repo = user_dir / name
        subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
        subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
        subprocess.run([*git, "commit", "-qm", "c"], cwd=repo, check=True)
        heads[name] = subprocess.run([*git, "rev-parse", "HEAD"], cwd=repo, check=True,
                                     capture_output=True, text=True).stdout.strip()
    pins = {"copy-same": "e" * 40, "copy-old": "e" * 40, "git-pinned": heads["git-pinned"], "git-off-pin": "a" * 40}
    for name, sha in pins.items():  # every catalog entry says version 0.1.3 (a git HEAD match outranks it)
        (catalog_dir / f"{name}.yaml").write_text(
            f"name: {name}\nrepo: https://github.com/example/{name}\nsha: {sha}\ndescription: d\n"
            f"maintainer: t\ncategory: tools\nversion: \"0.1.3\"\n", encoding="utf-8")
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir)
    monkeypatch.setattr("hermes_cli.plugin_catalog.get_catalog_dir", lambda: catalog_dir)
    monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.resolved_removed_entries", list)  # no network
    monkeypatch.setattr(importlib.metadata, "entry_points", list)

    plugins_cmd.cmd_list(_args(json=True))
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}

    assert {n: rows[n].get("update_available") for n in rows} == {
        "copy-same": False, "copy-old": True, "git-pinned": False, "git-off-pin": True, "not-in-catalog": None}
    assert rows["git-pinned"]["source"] == "git" and rows["copy-same"]["source"] == "user"
    assert set(rows["not-in-catalog"]) == {"name", "status", "version", "description", "source", "removed"}
