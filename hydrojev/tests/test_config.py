import json
from pathlib import Path

import pytest

from hydrojev.config import ConfigError, load_config


def test_default_config_resolves_reference_paths_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]

    config = load_config(project_root=project_root)

    assert config.paths.project_root == project_root
    assert config.paths.ref_code == project_root / "refCode"
    assert config.paths.artifacts == project_root / "artifacts"
    assert config.clustering.alpha + config.clustering.beta <= 1.0


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("clustering:\n  alpha: 0.8\n  beta: 0.4\n", r"alpha \+ beta"),
        ("clustering:\n  acoustic_speed_m_s: 0\n", "acoustic_speed_m_s"),
        ("concealment:\n  max_modified_features: 5\n", "max_modified_features"),
    ],
)
def test_invalid_safety_configuration_is_rejected(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(config_path=config_path, project_root=tmp_path)


def test_secret_cannot_be_loaded_or_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "secret.yaml"
    config_path.write_text("jev:\n  api_key: forbidden\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="API key"):
        load_config(config_path=config_path, project_root=tmp_path)

    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_test_should_not_serialize")
    public = load_config(project_root=Path(__file__).resolve().parents[2]).to_public_dict()
    serialized = json.dumps(public)
    assert "apikey_" not in serialized
    assert "TYPESAFE_API_KEY" not in serialized
