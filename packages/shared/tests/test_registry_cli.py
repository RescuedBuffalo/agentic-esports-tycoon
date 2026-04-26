"""``nexus run`` CLI surface — register / ls / show / finalize.

These tests drive the dispatcher (:func:`esports_sim.cli.main`) so they
exercise the same code path operators use, including argparse parsing
and exit codes. Each test isolates state via ``--db`` and ``--runs-dir``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.cli import main as cli_main


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Test sandbox: a config file + clean state/runs paths."""
    cfg = tmp_path / "configs" / "graph" / "era_7.09.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("kind: graph-snapshot\nera: 7.09\n", encoding="utf-8")
    return {
        "db": tmp_path / "state" / "registry.db",
        "runs_dir": tmp_path / "runs",
        "config": cfg,
    }


def _run_argv(workspace: dict[str, Path], *extra: str) -> list[str]:
    """Build a CLI argv with the workspace's --db / --runs-dir flags."""
    return [
        "run",
        "--db",
        str(workspace["db"]),
        "--runs-dir",
        str(workspace["runs_dir"]),
        *extra,
    ]


def test_register_prints_run_id_and_creates_artifacts(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The acceptance scenario from BUF-69:

    nexus run register --kind=graph-snapshot --config=...era_7.09.yaml
    → prints a run_id, creates runs/{id}/config.yaml + a DB row.
    """
    rc = cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=graph-snapshot",
            f"--config={workspace['config']}",
        )
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("graph-snapshot-")

    run_dir = workspace["runs_dir"] / out
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").read_bytes() == workspace["config"].read_bytes()


def test_register_is_idempotent_via_cli(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running the same command prints the same run_id (no-op)."""
    rc1 = cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=graph-snapshot",
            f"--config={workspace['config']}",
        )
    )
    first = capsys.readouterr().out.strip()

    rc2 = cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=graph-snapshot",
            f"--config={workspace['config']}",
        )
    )
    second = capsys.readouterr().out.strip()

    assert rc1 == rc2 == 0
    assert first == second


def test_ls_filters_by_kind_and_lists_status_and_duration(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """BUF-69 acceptance: ``nexus run ls --kind=rl-train`` lists runs with
    status + duration."""
    # Register two runs of different kinds.
    cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=graph-snapshot",
            f"--config={workspace['config']}",
        )
    )
    snap_id = capsys.readouterr().out.strip()

    # Need a different config to mint a distinct rl-train run.
    rl_config = workspace["config"].parent / "rl.yaml"
    rl_config.write_text("kind: rl\n", encoding="utf-8")
    cli_main(_run_argv(workspace, "register", "--kind=rl-train", f"--config={rl_config}"))
    rl_id = capsys.readouterr().out.strip()

    # Filter to rl-train: only the rl run shows up.
    rc = cli_main(_run_argv(workspace, "ls", "--kind=rl-train"))
    assert rc == 0
    out = capsys.readouterr().out
    assert rl_id in out
    assert snap_id not in out
    # Header columns expected by the issue (status + duration).
    assert "STATUS" in out
    assert "DUR" in out


def test_ls_filters_by_status(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cli_main(_run_argv(workspace, "register", "--kind=rl-train", f"--config={workspace['config']}"))
    run_id = capsys.readouterr().out.strip()
    cli_main(_run_argv(workspace, "finalize", run_id, "--status=completed"))

    rc = cli_main(_run_argv(workspace, "ls", "--status=running"))
    assert rc == 0
    assert capsys.readouterr().out == ""

    rc = cli_main(_run_argv(workspace, "ls", "--status=completed"))
    assert rc == 0
    assert run_id in capsys.readouterr().out


def test_show_prints_paths_and_metadata(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=rl-train",
            f"--config={workspace['config']}",
            "--notes=initial",
        )
    )
    run_id = capsys.readouterr().out.strip()

    rc = cli_main(_run_argv(workspace, "show", run_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert run_id in out
    # Both human-readable labels and the resolved paths are present —
    # this is what callers grep when they don't want to import the API.
    assert "config_snapshot" in out
    assert "config.yaml" in out
    assert "run_dir" in out
    assert "initial" in out


def test_show_unknown_run_id_returns_nonzero(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(_run_argv(workspace, "show", "nope-no-such-run"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "error:" in out.lower()


def test_finalize_marks_run_terminal(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cli_main(_run_argv(workspace, "register", "--kind=rl-train", f"--config={workspace['config']}"))
    run_id = capsys.readouterr().out.strip()

    rc = cli_main(_run_argv(workspace, "finalize", run_id, "--status=completed", "--notes=ok"))
    assert rc == 0

    # `show` reflects the new state.
    cli_main(_run_argv(workspace, "show", run_id))
    out = capsys.readouterr().out
    assert "completed" in out
    assert "duration" in out
    assert "ok" in out


def test_register_with_data_and_fingerprint_returns_error(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The CLI rejects the conflicting flags before touching the registry."""
    data = tmp_path / "data.csv"
    data.write_bytes(b"x")
    rc = cli_main(
        _run_argv(
            workspace,
            "register",
            "--kind=rl-train",
            f"--config={workspace['config']}",
            f"--data={data}",
            "--data-fingerprint=precomputed",
        )
    )
    assert rc == 2
    assert "not both" in capsys.readouterr().out


def test_top_level_help_lists_run_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Smoke test for the dispatcher: ``nexus -h`` mentions ``run``."""
    with pytest.raises(SystemExit):
        cli_main(["-h"])
    out = capsys.readouterr().out
    assert "run" in out
    assert "budget" in out
