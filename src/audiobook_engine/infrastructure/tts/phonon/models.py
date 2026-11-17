"""Phonon 40M model cache, first-use download and manual instructions.

Phonon 40M ships one checkpoint per language on Hugging Face
(``gradium/phonon-40m-<lang>``). Those repositories are gated, so a
first-use download needs the model license accepted and a Hugging Face
token. When the download cannot run, the user gets step-by-step
instructions including the exact local directory where the checkpoint
files must be placed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from audiobook_engine.domain.exceptions import TTSBackendError

SUPPORTED_LANGUAGES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
}

#: Hugging Face organization/repository prefix for the Phonon 40M models.
_HF_REPO_PREFIX = "gradium/phonon-40m-"
_HF_BASE_URL = "https://huggingface.co"

#: Files that must exist for a local checkpoint to be usable offline.
_REQUIRED_FILES: tuple[str, ...] = ("config.json", "tokenizer.json")

#: Weight files accepted by the runtime, in order of preference.
_WEIGHT_FILES: tuple[str, ...] = (
    "model.q8.gguf",
    "model.safetensors",
    "model.q4k.gguf",
)

#: Files fetched by the automatic first-use download, in order.
_DOWNLOAD_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "model.q8.gguf",
)

#: Optional files: a checkpoint without them still works.
_OPTIONAL_DOWNLOAD_FILES: tuple[str, ...] = ("default-voice.safetensors",)

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 1.0


class _DownloadError(Exception):
    """Internal download failure carrying a user-facing reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _DownloadNotFound(_DownloadError):
    """The requested file does not exist in the model repository."""


def normalize_phonon_language(language: str) -> str:
    """Normalize a language tag to a Phonon 40M language code.

    Regional tags are reduced to their primary subtag: ``en-US`` and
    ``en_GB`` become ``en``, ``pt-BR`` becomes ``pt``.

    Raises
    ------
    TTSBackendError
        If the language is not one of the five Phonon 40M languages.
    """
    raw = (language or "").strip().lower().replace("_", "-")
    primary = raw.split("-", 1)[0]
    if primary in SUPPORTED_LANGUAGES:
        return primary
    supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
    raise TTSBackendError(
        f"Language '{language}' is not supported by Phonon 40M. "
        f"Supported languages: {supported}"
    )


def try_normalize_phonon_language(language: str | None) -> str | None:
    """Return the normalized language, or ``None`` when unsupported."""
    if not language:
        return None
    raw = language.strip().lower().replace("_", "-")
    primary = raw.split("-", 1)[0]
    return primary if primary in SUPPORTED_LANGUAGES else None


def get_phonon_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the root directory holding the Phonon 40M checkpoints."""
    if cache_dir is not None:
        return Path(cache_dir)
    from platformdirs import user_cache_path

    return user_cache_path("ebook-audiobook-engine") / "phonon"


def get_language_model_dir(language: str, cache_dir: Path | None = None) -> Path:
    """Return the local directory expected for one language checkpoint."""
    lang = normalize_phonon_language(language)
    return get_phonon_cache_dir(cache_dir) / f"phonon-40m-{lang}"


def missing_checkpoint_files(model_dir: Path) -> list[str]:
    """List the required checkpoint files absent from ``model_dir``."""
    missing = [name for name in _REQUIRED_FILES if not (model_dir / name).is_file()]
    if not any((model_dir / name).is_file() for name in _WEIGHT_FILES):
        missing.append(_WEIGHT_FILES[0])
    return missing


def is_checkpoint_complete(model_dir: Path) -> bool:
    """Return True when ``model_dir`` holds a usable checkpoint."""
    return not missing_checkpoint_files(model_dir)


def list_cached_languages(cache_dir: Path | None = None) -> list[str]:
    """Return the languages whose checkpoint is complete on disk."""
    root = get_phonon_cache_dir(cache_dir)
    if not root.is_dir():
        return []
    return [
        lang
        for lang in sorted(SUPPORTED_LANGUAGES)
        if is_checkpoint_complete(root / f"phonon-40m-{lang}")
    ]


def detect_quant(model_dir: Path) -> str | None:
    """Return the weight format name for the files present in ``model_dir``."""
    if (model_dir / "model.q8.gguf").is_file():
        return "q8"
    if (model_dir / "model.q4k.gguf").is_file():
        return "q4k"
    return None


def _hf_home() -> Path:
    """Return the Hugging Face home directory, as the runtime reads it."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "huggingface"


def _resolve_hf_token() -> str | None:
    """Return the Hugging Face token from the environment or stored login."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_path = os.environ.get("HF_TOKEN_PATH")
    path = Path(token_path) if token_path else _hf_home() / "token"
    try:
        stored = path.read_text().strip()
    except OSError:
        return None
    return stored or None


def _create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "ebook-audiobook-engine/1.0",
            "Accept": "application/octet-stream",
        }
    )
    return session


def _download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    token: str | None,
) -> None:
    """Download ``url`` into ``dest`` atomically."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = session.get(url, stream=True, timeout=30, headers=headers)
    except requests.exceptions.SSLError as exc:
        raise _DownloadError(
            "SSL error while contacting Hugging Face. Check your network, "
            "proxy or certificate store."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise _DownloadError(
            "Connection error while contacting Hugging Face. "
            "Check your internet connection."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise _DownloadError(
            "Timeout while contacting Hugging Face. Check your internet connection."
        ) from exc

    if response.status_code == 404:
        raise _DownloadNotFound(f"file not found in the model repository: {url}")
    if response.status_code in (401, 403):
        raise _DownloadError(
            "access denied by Hugging Face (HTTP "
            f"{response.status_code}). The model is gated: accept its license "
            "and provide a token."
        )
    if response.status_code >= 400:
        raise _DownloadError(f"HTTP {response.status_code} while downloading {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        if tmp_path.stat().st_size == 0:
            raise _DownloadError(f"downloaded an empty file: {url}")
        shutil.move(str(tmp_path), str(dest))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_phonon_checkpoint(language: str, model_dir: Path) -> None:
    """Download the checkpoint files for ``language`` into ``model_dir``."""
    repo_url = f"{_HF_BASE_URL}/{_HF_REPO_PREFIX}{language}"
    token = _resolve_hf_token()
    session = _create_session()
    created: list[Path] = []

    try:
        for name in _DOWNLOAD_FILES:
            dest = model_dir / name
            _download_file(session, f"{repo_url}/resolve/main/{name}", dest, token)
            created.append(dest)
        for name in _OPTIONAL_DOWNLOAD_FILES:
            dest = model_dir / name
            try:
                _download_file(session, f"{repo_url}/resolve/main/{name}", dest, token)
            except _DownloadNotFound:
                continue
            created.append(dest)
    except _DownloadError:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _format_download_instructions(
    language: str,
    model_dir: Path,
    reason: str,
    missing_files: list[str] | None = None,
) -> str:
    """Build the user-facing instructions for a missing checkpoint."""
    repo_url = f"{_HF_BASE_URL}/{_HF_REPO_PREFIX}{language}"
    target = model_dir.resolve()
    lines = [
        f"Could not prepare the Phonon 40M model for language '{language}' "
        f"({_HF_REPO_PREFIX}{language}).",
        f"Reason: {reason}",
    ]
    if missing_files:
        lines.append(f"Missing files: {', '.join(missing_files)}")
    lines.extend(
        [
            "",
            "Phonon 40M models on Hugging Face require accepting the model license.",
            "",
            "Option 1 - Automatic download on first use:",
            f"  1. Accept the license at: {repo_url}",
            "  2. Authenticate with Hugging Face:",
            "     export HF_TOKEN=hf_...   (or run: hf auth login)",
            "  3. Re-run the command.",
            "",
            "Option 2 - Manual download:",
            f"  1. Download these files from {repo_url}/tree/main:",
            "     - config.json",
            "     - tokenizer.json",
            "     - model.q8.gguf (or model.safetensors)",
            "     - default-voice.safetensors (optional, plus any "
            "voices/*.safetensors)",
            "  2. Place them in this exact local directory:",
            f"     {target}",
        ]
    )
    return "\n".join(lines)


def ensure_model_dir(
    language: str,
    cache_dir: Path | None = None,
    auto_download: bool = True,
) -> Path:
    """Ensure the checkpoint for ``language`` is available locally.

    Returns the directory holding a usable checkpoint. A complete local
    checkpoint is used without any network access.
    """
    lang = normalize_phonon_language(language)
    model_dir = get_language_model_dir(lang, cache_dir)

    missing = missing_checkpoint_files(model_dir)
    if not missing:
        return model_dir

    partially_populated = model_dir.is_dir() and any(model_dir.iterdir())
    if partially_populated:
        raise TTSBackendError(
            _format_download_instructions(
                lang,
                model_dir,
                reason="the local model directory exists but is incomplete",
                missing_files=missing,
            )
        )

    if not auto_download:
        raise TTSBackendError(
            _format_download_instructions(
                lang,
                model_dir,
                reason="the model is not present locally",
                missing_files=missing,
            )
        )

    try:
        _download_phonon_checkpoint(lang, model_dir)
    except _DownloadError as exc:
        raise TTSBackendError(
            _format_download_instructions(
                lang,
                model_dir,
                reason=exc.reason,
                missing_files=missing,
            )
        ) from exc

    missing = missing_checkpoint_files(model_dir)
    if missing:
        raise TTSBackendError(
            _format_download_instructions(
                lang,
                model_dir,
                reason="the download finished but the checkpoint is incomplete",
                missing_files=missing,
            )
        )
    return model_dir
