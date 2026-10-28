import json
import zipfile

import pytest

from realtime.scripts.demo_assets import MANIFEST, check, export_assets, import_assets


def test_export_is_allowlisted_and_roundtrip_matches(demo_assets, tmp_path):
    (demo_assets / ".env").write_text("PRIVATE=must-not-be-shared")
    (demo_assets / "unrelated.txt").write_text("not an input")
    archive = tmp_path / "assets.zip"
    report = export_assets(demo_assets, archive)
    assert report["media_files"] == 9
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == 10
        assert (
            ".env" not in bundle.namelist() and "unrelated.txt" not in bundle.namelist()
        )
    imported = tmp_path / "imported"
    assert import_assets(archive, imported)["verified"]
    assert check(imported) == check(demo_assets)
    with pytest.raises(FileExistsError):
        export_assets(demo_assets, archive)
    with pytest.raises(FileExistsError):
        import_assets(archive, imported)


def test_tampered_pack_is_not_installed(demo_assets, tmp_path):
    archive = tmp_path / "bad.zip"
    record = check(demo_assets)
    record["files"]["images/ref01.png"]["sha256"] = "0" * 64
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in record["files"]:
            bundle.write(demo_assets / name, name)
        bundle.writestr(MANIFEST, json.dumps(record))
    target = tmp_path / "destination"
    with pytest.raises(ValueError, match="manifest"):
        import_assets(archive, target)
    assert not target.exists()


def test_archive_paths_cannot_escape_destination(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "not allowed")
    with pytest.raises(ValueError, match="exactly"):
        import_assets(archive, tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()


def test_missing_inputs_have_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="import the demo asset pack"):
        check(tmp_path)
