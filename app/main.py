import base64
import os
from typing import Any

import easyocr
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI(title="easy-ocr-api")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error - usa /ocr para JSON con image_base64, "
            "o /ocr/upload para multipart file"
        },
    )


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


def _rotate_image(img: Image.Image, degrees: int) -> Image.Image:
    """Rota la imagen los grados especificados (sentido antihorario)."""
    if degrees == 0:
        return img
    return img.rotate(degrees, expand=True)


# Inicializa una sola vez; la primera carga puede tardar y descargar modelos.
_READER = easyocr.Reader(["es", "en"], gpu=False)

# Threshold minimo de confianza para aceptar resultado
_CONFIDENCE_THRESHOLD = 0.7


def _run_ocr_on_image(img: Image.Image) -> tuple[list[dict[str, Any]], float]:
    """Ejecuta OCR y devuelve (lines, avg_confidence)."""
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

    avg_confidence = (
        float(sum(line["confidence"] for line in lines) / len(lines)) if lines else 0.0
    )
    return lines, avg_confidence


async def _process_ocr(image_bytes: bytes) -> dict[str, Any]:
    """Procesa la imagen probando rotaciones hasta obtener confianza >= 0.7."""
    img = _load_image_bytes(image_bytes)

    rotations = [0, 90, 180, 270]
    best_result: dict[str, Any] | None = None
    best_confidence = 0.0

    for degrees in rotations:
        rotated_img = _rotate_image(img, degrees)
        lines, avg_confidence = _run_ocr_on_image(rotated_img)

        # Si supera el threshold, usar este resultado
        if avg_confidence >= _CONFIDENCE_THRESHOLD:
            return {
                "full_text": "\n".join([line["text"] for line in lines]).strip(),
                "avg_confidence": avg_confidence,
                "lines": lines,
                "rotated_degrees": degrees,
                "low_confidence": False,
                "languages": ["es", "en"],
                "gpu": False,
            }

        # Guardar el mejor resultado por si ninguno supera el threshold
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_result = {
                "full_text": "\n".join([line["text"] for line in lines]).strip(),
                "avg_confidence": avg_confidence,
                "lines": lines,
                "rotated_degrees": degrees,
                "low_confidence": True,
                "languages": ["es", "en"],
                "gpu": False,
            }

    # Ninguna rotacion supero el threshold, devolver la mejor con flag
    return best_result or {
        "full_text": "",
        "avg_confidence": 0.0,
        "lines": [],
        "rotated_degrees": 0,
        "low_confidence": True,
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
