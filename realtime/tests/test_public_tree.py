import subprocess

from scripts.check_public_tree import check


def test_public_check_includes_force_tracked_private_files(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / ".env").write_text("PRIVATE=fixture-value\n")
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n")
    assert check(tmp_path) == []
    subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True)
    assert check(tmp_path) == [".env: generated/private file would be published"]


def test_public_check_reports_location_without_echoing_credential(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    credential = "sk-" + "a" * 40
    (tmp_path / "bad.py").write_text(f"key = '{credential}'\n")
    failures = check(tmp_path)
    assert failures == ["bad.py:1: possible OpenAI credential"]
    assert credential not in str(failures)
