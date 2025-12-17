import base64
import os
from typing import Any

import easyocr
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI(title="easy-ocr-api")


class OCRBase64Request(BaseModel):
    image_base64: str


def _get_api_token() -> str:
    token = os.getenv("API_TOKEN")
    if not token:
        raise RuntimeError("API_TOKEN no está configurado en el entorno")
    return token


def require_auth(
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    expected = _get_api_token()

    bearer: str | None = None
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            bearer = parts[1].strip()

    provided = x_api_token or bearer
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _decode_base64_image(data: str) -> bytes:
    # Acepta "data:image/...;base64,...." o base64 plano
    if "," in data and data.strip().lower().startswith("data:"):
        _, data = data.split(",", 1)

    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Base64 inválido") from exc


def _load_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Imagen inválida") from exc


def _rotate_90(img: Image.Image) -> Image.Image:
    # Rotación fija 90 grados (PIL: sentido antihorario)
    return img.rotate(90, expand=True)


# Inicializa una sola vez; la primera carga puede tardar y descargar modelos.
_READER = easyocr.Reader(["es", "en"], gpu=False)


async def _process_ocr(image_bytes: bytes) -> dict[str, Any]:
    """Procesa la imagen y retorna el resultado OCR."""
    img = _load_image_bytes(image_bytes)
    img = _rotate_90(img)

    img_np = np.array(img)
    results = _READER.readtext(img_np)

    lines: list[dict[str, Any]] = []
    for bbox, text, confidence in results:
        lines.append(
            {
                "text": text,
                "confidence": float(confidence),
                "bbox": [[float(x), float(y)] for x, y in bbox],
            }
        )

    full_text = "\n".join([line["text"] for line in lines]).strip()
    avg_confidence = (
        float(sum(line["confidence"] for line in lines) / len(lines)) if lines else 0.0
    )

    return {
        "full_text": full_text,
        "avg_confidence": avg_confidence,
        "lines": lines,
        "rotated_degrees": 90,
        "languages": ["es", "en"],
        "gpu": False,
    }


@app.post("/ocr")
async def ocr_json(
    payload: OCRBase64Request,
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    """OCR via JSON con image_base64."""
    image_bytes = _decode_base64_image(payload.image_base64)
    return await _process_ocr(image_bytes)


@app.post("/ocr/upload")
async def ocr_upload(
    _: None = Depends(require_auth),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """OCR via multipart file upload."""
    image_bytes = await file.read()
    return await _process_ocr(image_bytes)
