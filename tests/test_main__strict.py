import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image


def _make_png_bytes(*, size: tuple[int, int] = (2, 3)) -> bytes:
    if not isinstance(size, tuple) or len(size) != 2:
        raise TypeError("size must be tuple[int, int]")
    if any(type(x) is not int for x in size):
        raise TypeError("size must be tuple[int, int]")

    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    out = buf.getvalue()
    assert type(out) is bytes
    assert len(out) > 0
    return out


@pytest.fixture()
def app_main():
    # Import inside fixture so tests always go through our conftest easyocr stub.
    import app.main as main

    # Ensure our easyocr module is the stub, and _READER exists.
    import easyocr as easyocr_module

    assert hasattr(main, "_READER")
    assert isinstance(main._READER, easyocr_module.Reader)
    assert type(main._READER.init_languages) is list
    assert main._READER.init_languages == ["es", "en"]
    assert type(main._READER.init_gpu) is bool
    assert main._READER.init_gpu is False

    main._READER.reset()
    return main


@pytest.fixture()
def api_token(monkeypatch: pytest.MonkeyPatch) -> str:
    token = "test-token"
    monkeypatch.setenv("API_TOKEN", token)
    return token


@pytest.mark.asyncio
async def test_post_ocr__auth_missing_headers__401(app_main, api_token: str) -> None:
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ocr")

    assert resp.status_code == 401
    body = resp.json()
    assert type(body) is dict
    assert body == {"detail": "Unauthorized"}


@pytest.mark.asyncio
async def test_post_ocr__auth_valid_x_api_token__200(app_main, api_token: str) -> None:
    # Return empty OCR results for deterministic output.
    app_main._READER.set_readtext_return([])

    img_bytes = _make_png_bytes()

    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"X-API-Token": api_token},
            files={"file": ("img.png", img_bytes, "image/png")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert type(body) is dict

    assert set(body.keys()) == {
        "full_text",
        "avg_confidence",
        "lines",
        "rotated_degrees",
        "languages",
        "gpu",
    }

    assert type(body["full_text"]) is str
    assert body["full_text"] == ""

    assert type(body["avg_confidence"]) is float
    assert body["avg_confidence"] == 0.0

    assert type(body["lines"]) is list
    assert body["lines"] == []

    assert type(body["rotated_degrees"]) is int
    assert body["rotated_degrees"] == 90

    assert type(body["languages"]) is list
    assert body["languages"] == ["es", "en"]

    assert type(body["gpu"]) is bool
    assert body["gpu"] is False


@pytest.mark.asyncio
async def test_post_ocr__auth_valid_bearer_token_case_insensitive__200(
    app_main, api_token: str
) -> None:
    app_main._READER.set_readtext_return([])

    img_bytes = _make_png_bytes()

    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"Authorization": f"bEaReR    {api_token}  "},
            files={"file": ("img.png", img_bytes, "image/png")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert type(body) is dict
    assert body["languages"] == ["es", "en"]


@pytest.mark.asyncio
async def test_post_ocr__auth_invalid_token__401(app_main, api_token: str) -> None:
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"X-API-Token": "wrong"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert type(body) is dict
    assert body == {"detail": "Unauthorized"}


@pytest.mark.asyncio
async def test_post_ocr__auth_x_api_token_takes_precedence_over_bearer__401(
    app_main, api_token: str
) -> None:
    # Contract: provided = x_api_token or bearer, so a wrong X-API-Token must fail even if bearer is valid.
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={
                "X-API-Token": "wrong",
                "Authorization": f"Bearer {api_token}",
            },
        )

    assert resp.status_code == 401
    body = resp.json()
    assert type(body) is dict
    assert body == {"detail": "Unauthorized"}


def test__get_api_token__missing_env_raises_runtime_error(
    app_main, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        app_main._get_api_token()

    assert type(excinfo.value) is RuntimeError
    assert str(excinfo.value) == "API_TOKEN no está configurado en el entorno"


def test_require_auth__valid_x_api_token_returns_none(app_main, api_token: str) -> None:
    out = app_main.require_auth(x_api_token=api_token, authorization=None)
    assert out is None


def test_require_auth__valid_bearer_returns_none(app_main, api_token: str) -> None:
    out = app_main.require_auth(x_api_token=None, authorization=f"Bearer {api_token}")
    assert out is None


def test_require_auth__invalid_or_missing_raises_http_exception(
    app_main, api_token: str
) -> None:
    for x_api_token, authorization in [
        (None, None),
        ("", None),
        ("wrong", None),
        (None, "Bearer wrong"),
        (None, "Token test-token"),  # not bearer scheme
        (None, "Bearer"),  # no token part
        (None, "Bearer "),  # empty token
    ]:
        with pytest.raises(HTTPException) as excinfo:
            app_main.require_auth(x_api_token=x_api_token, authorization=authorization)

        exc = excinfo.value
        assert type(exc) is HTTPException
        assert exc.status_code == 401
        assert exc.detail == "Unauthorized"


def test__decode_base64_image__plain_base64_returns_exact_bytes(app_main) -> None:
    raw = b"abc\x00\xff"
    b64 = base64.b64encode(raw).decode("ascii")
    out = app_main._decode_base64_image(b64)

    assert type(out) is bytes
    assert out == raw


def test__decode_base64_image__data_url_base64_returns_exact_bytes(app_main) -> None:
    raw = _make_png_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    out = app_main._decode_base64_image(data_url)

    assert type(out) is bytes
    assert out == raw


def test__decode_base64_image__empty_string_is_valid_base64_and_returns_empty_bytes(
    app_main,
) -> None:
    # Contract: base64.b64decode('', validate=True) returns b''.
    out = app_main._decode_base64_image("")
    assert type(out) is bytes
    assert out == b""


def test__decode_base64_image__invalid_base64_raises_http_exception(app_main) -> None:
    with pytest.raises(HTTPException) as excinfo:
        app_main._decode_base64_image("not_base64!!")

    exc = excinfo.value
    assert type(exc) is HTTPException
    assert exc.status_code == 400
    assert exc.detail == "Base64 inválido"


def test__load_image_bytes__valid_png_returns_rgb_image(app_main) -> None:
    img_bytes = _make_png_bytes()
    img = app_main._load_image_bytes(img_bytes)

    assert isinstance(img, Image.Image)
    assert type(img.mode) is str
    assert img.mode == "RGB"  # convert("RGB") is part of the contract


def test__load_image_bytes__invalid_bytes_raises_http_exception(app_main) -> None:
    with pytest.raises(HTTPException) as excinfo:
        app_main._load_image_bytes(b"not an image")

    exc = excinfo.value
    assert type(exc) is HTTPException
    assert exc.status_code == 400
    assert exc.detail == "Imagen inválida"


def test__rotate_90__swaps_dimensions_and_rotates_pixels(app_main) -> None:
    # 2x3 image with unique colors per pixel to test exact rotation mapping.
    img = Image.new("RGB", (2, 3))
    img.putpixel((0, 0), (1, 2, 3))
    img.putpixel((1, 0), (4, 5, 6))
    img.putpixel((0, 1), (7, 8, 9))
    img.putpixel((1, 1), (10, 11, 12))
    img.putpixel((0, 2), (13, 14, 15))
    img.putpixel((1, 2), (16, 17, 18))

    rotated = app_main._rotate_90(img)

    assert isinstance(rotated, Image.Image)
    assert rotated.size == (3, 2)  # expand=True swaps width/height

    # CCW 90 mapping: (x, y) -> (y, W-1-x) where W is original width.
    # Original W=2.
    assert rotated.getpixel((0, 1)) == (1, 2, 3)  # (0,0)->(0,1)
    assert rotated.getpixel((0, 0)) == (4, 5, 6)  # (1,0)->(0,0)
    assert rotated.getpixel((2, 1)) == (13, 14, 15)  # (0,2)->(2,1)


@pytest.mark.asyncio
async def test_post_ocr__input_missing_image__400(app_main, api_token: str) -> None:
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ocr", headers={"X-API-Token": api_token})

    assert resp.status_code == 400
    body = resp.json()
    assert type(body) is dict
    assert body == {
        "detail": "Debes enviar multipart con 'file' o JSON con 'image_base64'"
    }


@pytest.mark.asyncio
async def test_post_ocr__input_invalid_base64__400(app_main, api_token: str) -> None:
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"X-API-Token": api_token, "Content-Type": "application/json"},
            json={"image_base64": "not_base64!!"},
        )

    assert resp.status_code == 400
    body = resp.json()
    assert type(body) is dict
    assert body == {"detail": "Base64 inválido"}


@pytest.mark.asyncio
async def test_post_ocr__input_invalid_image_bytes__400(
    app_main, api_token: str
) -> None:
    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"X-API-Token": api_token},
            files={"file": ("img.png", b"not an image", "image/png")},
        )

    assert resp.status_code == 400
    body = resp.json()
    assert type(body) is dict
    assert body == {"detail": "Imagen inválida"}


@pytest.mark.asyncio
async def test_post_ocr__json_base64_success_and_strict_response_contract(
    app_main, api_token: str
) -> None:
    # Non-square image to assert rotation via numpy array shape.
    img_bytes = _make_png_bytes(size=(2, 3))  # W=2, H=3

    bbox_1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    bbox_2 = [(2, 2), (3, 2), (3, 3), (2, 3)]
    app_main._READER.set_readtext_return(
        [
            (bbox_1, "hola", np.float32(0.5)),
            (bbox_2, "mundo ", 1.0),
        ]
    )

    payload = {"image_base64": base64.b64encode(img_bytes).decode("ascii")}

    transport = ASGITransport(app=app_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr",
            headers={"X-API-Token": api_token, "Content-Type": "application/json"},
            json=payload,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert type(body) is dict

    assert set(body.keys()) == {
        "full_text",
        "avg_confidence",
        "lines",
        "rotated_degrees",
        "languages",
        "gpu",
    }

    assert body["full_text"] == "hola\nmundo"  # trailing space stripped by .strip()
    assert type(body["full_text"]) is str

    assert type(body["avg_confidence"]) is float
    assert body["avg_confidence"] == 0.75

    assert type(body["lines"]) is list
    assert len(body["lines"]) == 2

    for line in body["lines"]:
        assert type(line) is dict
        assert set(line.keys()) == {"text", "confidence", "bbox"}
        assert type(line["text"]) is str
        assert type(line["confidence"]) is float

        bbox = line["bbox"]
        assert type(bbox) is list
        assert len(bbox) == 4
        for pt in bbox:
            assert type(pt) is list
            assert pt.__class__ is list  # avoid subclasses
            assert len(pt) == 2
            assert type(pt[0]) is float
            assert type(pt[1]) is float

    assert body["lines"][0]["text"] == "hola"
    assert body["lines"][0]["confidence"] == 0.5
    assert body["lines"][0]["bbox"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    assert body["lines"][1]["text"] == "mundo "
    assert body["lines"][1]["confidence"] == 1.0
    assert body["lines"][1]["bbox"] == [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]

    assert body["rotated_degrees"] == 90
    assert type(body["rotated_degrees"]) is int

    assert body["languages"] == ["es", "en"]
    assert type(body["languages"]) is list

    assert body["gpu"] is False
    assert type(body["gpu"]) is bool

    # Strictly assert EasyOCR readtext call contract and rotation effect.
    assert type(app_main._READER.readtext_calls) is list
    assert len(app_main._READER.readtext_calls) == 1
    img_np = app_main._READER.readtext_calls[0]
    assert isinstance(img_np, np.ndarray)
    assert img_np.dtype == np.uint8

    # Original image W=2, H=3; after rotate 90 CCW: PIL size (3,2) -> numpy shape (2,3,3).
    assert img_np.shape == (2, 3, 3)


@pytest.mark.asyncio
async def test_ocr_function__rejects_both_file_and_payload__400(
    app_main, api_token: str
) -> None:
    # This branch is hard to hit via HTTP because a request can't be both JSON-body and multipart.
    from fastapi import UploadFile

    img_bytes = _make_png_bytes()
    upload = UploadFile(filename="img.png", file=BytesIO(img_bytes))
    payload = app_main.OCRBase64Request(
        image_base64=base64.b64encode(img_bytes).decode("ascii")
    )

    with pytest.raises(HTTPException) as excinfo:
        await app_main.ocr(_=None, file=upload, payload=payload)

    exc = excinfo.value
    assert type(exc) is HTTPException
    assert exc.status_code == 400
    assert exc.detail == "Envía solo uno: multipart 'file' o JSON 'image_base64'"
