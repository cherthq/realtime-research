from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / "realtime" / ".env", extra="ignore"
    )

    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_browser_url: str | None = None
    livekit_api_key: str = "devkey"
    livekit_api_secret: SecretStr = SecretStr("secret")
    openai_api_key: SecretStr = SecretStr("")
    openai_realtime_model: str = "gpt-realtime"
    openai_casting_model: str = "gpt-5.6-terra"
    state_dir: Path = ROOT / "data" / "realtime"
    demo_access_token: SecretStr = SecretStr("")
    control_transport: Literal["sse", "poll"] = "sse"
    assets_dir: Path = ROOT / "data" / "eval" / "facetime-zero-shot-v1"
    session_limit_seconds: float = 600
    reconnect_grace_seconds: float = 30
    avatarforever_enabled: bool = False
    avatarforever_compile_blocks: bool = False
    avatarforever_python: Path = ROOT / ".venv-avatarforever/bin/python"
    avatarforever_weights: Path = ROOT / "checkpoints/avatarforever"
    gpu_load_timeout_seconds: float = 300
    gpu_chunk_timeout_seconds: float = 60


settings = Settings()
