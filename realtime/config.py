from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    tts_provider: Literal["openai", "qwen", "elevenlabs"] = "openai"
    web_search_enabled: bool = False
    web_search_model: str = "gpt-4.1-mini"
    web_search_timeout_seconds: float = Field(default=8, ge=1, le=30)
    web_search_cache_seconds: float = Field(default=60, ge=0, le=300)
    elevenlabs_api_key: SecretStr = SecretStr("")
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_v4_turbo"
    # None preserves the provider's voice default. Only stability is wired for
    # dialogue WS; settings from other endpoints are not assumed compatible.
    elevenlabs_stability: float | None = Field(default=None, ge=0, le=1)
    elevenlabs_phrase_wait_ms: int = Field(default=120, ge=0, le=250)
    input_transcription_model: str = "gpt-live-transcribe"
    input_transcription_language: str = "en"
    camera_vision: bool = True
    qwen_tts_url: str = "http://127.0.0.1:18920"
    qwen_tts_timeout_seconds: float = 60
    voice_silence_duration_ms: int = Field(default=350, ge=200, le=2000)
    openai_casting_model: str = "gpt-5.6-terra"
    state_dir: Path = ROOT / "data" / "realtime"
    avatar_catalog: Literal["demo", "research"] = "demo"
    demo_access_token: SecretStr = SecretStr("")
    control_transport: Literal["sse", "poll"] = "sse"
    assets_dir: Path = ROOT / "data" / "eval" / "facetime-zero-shot-v1"
    session_limit_seconds: float = 600
    reconnect_grace_seconds: float = 30
    avatarforever_enabled: bool = False
    avatarforever_compile_blocks: bool = False
    avatarforever_cache_prompt: bool = False
    avatarforever_listening_profile: Literal["off", "onset_boost"] = "off"
    avatar_idle_motion: bool = False
    avatar_continuous_motion: bool = True
    avatar_startup_buffer_ms: int = Field(default=480, ge=0, le=1280)
    avatarforever_python: Path = ROOT / ".venv-avatarforever/bin/python"
    avatarforever_weights: Path = ROOT / "checkpoints/avatarforever"
    gpu_load_timeout_seconds: float = 300
    gpu_chunk_timeout_seconds: float = 60


settings = Settings()
