"""Model download and cache management for Kokoro TTS backend.

Models are downloaded from GitHub releases on first use and cached
locally. They are never committed to the repository.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Model file URLs (GitHub releases)
_KOKORO_ONNX_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx"
)
_VOICES_BIN_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin"
)

_KOKORO_ONNX_SIZE = 310_500_000  # ~310 MB
_VOICES_BIN_SIZE = 27_000_000  # ~27 MB

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 1.0


def _default_cache_dir() -> Path:
    """Return the default cache directory for Kokoro models."""
    from platformdirs import user_cache_path

    return user_cache_path("ebook-audiobook-engine") / "kokoro"


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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
    session.headers.update({
        "User-Agent": "ebook-audiobook-engine/1.0",
        "Accept": "application/octet-stream",
    })
    return session


def _download_file(url: str, dest: Path, expected_size: int) -> None:
    """Download a file with progress indication to a temp location, then move."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        print(f"Downloading {dest.name}...")
        session = _create_session()

        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = min(downloaded * 100 // total, 100)
                        if percent % 10 == 0:
                            mb = downloaded // 1024 // 1024
                            print(
                                f"\r  {percent}% ({mb} MB)...",
                                end="",
                                flush=True,
                            )

        print()  # newline after progress

        actual_size = tmp_path.stat().st_size
        if actual_size < expected_size * 0.9:
            raise RuntimeError(
                f"Downloaded file too small: {actual_size} bytes "
                f"(expected ~{expected_size})"
            )
        shutil.move(str(tmp_path), str(dest))
        print(f"Saved to {dest}")

    except requests.exceptions.SSLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SSL error downloading {dest.name}. "
            f"This may be caused by a firewall, proxy, or "
            f"outdated SSL certificates. Try:\n"
            f"  1. Check your network connection\n"
            f"  2. Update certificates: pip install --upgrade certifi\n"
            f"  3. Download manually:\n"
            f"     curl -L {url} -o {dest}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Connection error downloading {dest.name}. "
            f"Check your internet connection and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Timeout downloading {dest.name}. "
            f"Check your internet connection and try again."
        ) from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def get_model_path(cache_dir: Path | None = None) -> Path:
    """Ensure the Kokoro ONNX model is available locally.

    Downloads from GitHub releases if not cached.
    Returns the path to kokoro-v1.0.onnx.
    """
    cache = cache_dir or _default_cache_dir()
    model_path = cache / "kokoro-v1.0.onnx"

    if not model_path.exists():
        _download_file(_KOKORO_ONNX_URL, model_path, _KOKORO_ONNX_SIZE)

    return model_path


def get_voices_path(cache_dir: Path | None = None) -> Path:
    """Ensure the Kokoro voices file is available locally.

    Downloads from GitHub releases if not cached.
    Returns the path to voices-v1.0.bin.
    """
    cache = cache_dir or _default_cache_dir()
    voices_path = cache / "voices-v1.0.bin"

    if not voices_path.exists():
        _download_file(_VOICES_BIN_URL, voices_path, _VOICES_BIN_SIZE)

    return voices_path
