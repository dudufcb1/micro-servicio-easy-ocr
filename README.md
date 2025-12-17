# Microservicio EasyOCR (FastAPI)

Microservicio de OCR listo para desplegar con Docker.

## Requisitos

- Docker
- Docker Compose

## Configuración

1. Copia el archivo de ejemplo:

```bash
cp .env.example .env
```

2. Edita `.env` y define tu token:

```dotenv
API_TOKEN=tu_token
```

## Levantar el servicio

```bash
docker compose up --build
```

El API queda en `http://localhost:8000`.

## Autenticación

Envía el token en:

- `X-API-Token: <token>`

o

- `Authorization: Bearer <token>`

## Uso

### 1) Multipart (recomendado)

```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "X-API-Token: tu_token" \
  -F "file=@/ruta/a/imagen.jpg"
```

### 2) Base64 (JSON)

```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "X-API-Token: tu_token" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<BASE64_AQUI>"}'
```

## Respuesta

Devuelve JSON con:

- `full_text`: texto completo (concatenado por líneas)
- `lines`: lista de líneas con `text`, `confidence` y `bbox`

## Notas

- La imagen se rota 90 grados antes del OCR.
- EasyOCR puede descargar modelos la primera vez que arranca (dependiendo de si ya existen en la imagen/contenedor).
