import sys
import types
from pathlib import Path
from typing import Any


# Ensure project root is importable regardless of current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)


class _StubEasyOCRReader:
    """Stub that replaces easyocr.Reader to avoid heavy imports/models."""

    def __init__(self, languages: list[str], gpu: bool = False) -> None:
        if not isinstance(languages, list):
            raise TypeError("languages must be list[str]")
        if any(type(x) is not str for x in languages):
            raise TypeError("languages must be list[str]")
        if type(gpu) is not bool:
            raise TypeError("gpu must be bool")

        self.init_languages: list[str] = languages
        self.init_gpu: bool = gpu
        self.readtext_calls: list[Any] = []
        self._readtext_return: list[Any] = []

    def reset(self) -> None:
        self.readtext_calls.clear()
        self._readtext_return = []

    def set_readtext_return(self, value: list[Any]) -> None:
        if not isinstance(value, list):
            raise TypeError("readtext return must be a list")
        self._readtext_return = value

    def readtext(self, img_np: Any) -> list[Any]:
        # Capture exact object passed for strict assertions.
        self.readtext_calls.append(img_np)
        return self._readtext_return


# Inject stub module before importing app.main (main.py creates a module-global _READER).
_easyocr_stub = types.ModuleType("easyocr")
_easyocr_stub.Reader = _StubEasyOCRReader  # type: ignore[attr-defined]
sys.modules["easyocr"] = _easyocr_stub
