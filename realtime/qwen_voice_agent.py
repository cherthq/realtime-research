"""Compatibility imports for the original Qwen voice adapter."""

from .text_voice_agent import TextPhrases, TextVoiceAgent, speech_text

QwenVoiceAgent = TextVoiceAgent
__all__ = ["QwenVoiceAgent", "TextPhrases", "speech_text"]
