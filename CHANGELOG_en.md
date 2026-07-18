# Changelog

## v1.3.0 (2026-07-18)
- The Settings screen title now shows the version number (e.g., Nasu Voice Settings - Ver.1.3.0)
- The version number is now managed in a single place (`version.py`), referenced by both the Settings screen and the installer

## v1.2.0 (2026-07-17)
- Settings screen now supports Japanese and English (follows the language selected in the installer)
- Added a note on the installer's Welcome screen that the selected language also applies to the Settings screen
- Added a Show/Hide toggle next to the Groq API key field; the Change API Key dialog now pre-fills the existing key
- Fixed the shortcut key "Change" button not working (a build dependency had been missing since v1.1.1)

## v1.1.2 (2026-07-16)
- Fixed a crash when launching the Settings screen alone with no config file (nasu_config.json) present (added the same auto-create-if-missing logic used by the main app)

## v1.1.1 (2026-07-14)
- Preventive fix for a possible stuck Ctrl key during paste (guaranteed key release via try/finally)
- Fixed the Select Additional Tasks screen showing Japanese text even when English was selected

## v1.1.0 (2026-06-29)
- Installer now supports both Japanese and English (language selectable at startup)
- Added troubleshooting steps to README for download/install issues

## v1.0.0 (2026-06-19)
- Initial release of Windows voice input tool
- Voice transcription via Groq Whisper API
- Toggle / Push-to-Talk mode switching
- Customizable shortcut key
- Language selection (99 languages supported)
- Custom dictionary (up to 30 words)
- LLM-based correction (auto-fix misrecognition, toggle on/off)
- Installer (Japanese only)
