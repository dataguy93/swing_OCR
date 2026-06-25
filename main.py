import base64
import json
import os
import tempfile
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

from ocr_engine import ocr_scorecard

PLACES_AUTOCOMPLETE_URL = "https://places.googleapis.com/v1/places:autocomplete"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/ocr", methods=["OPTIONS"])
def ocr_options():
    return ("", 204)


def _decode_base64_image(image_base64):
    if not image_base64:
        return None
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    return base64.b64decode(image_base64)


def _create_temp_image_file(image_file=None, image_base64=None):
    if image_file is None and not image_base64:
        raise ValueError("No image content was provided")

    suffix = ".jpg"
    if image_file is not None:
        _, ext = os.path.splitext(image_file.filename or "")
        if ext:
            suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp)
            return tmp.name

    image_bytes = _decode_base64_image(image_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        return tmp.name


@app.get("/")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/ready")
def ready_check():
    return jsonify({"ready": True, "service": "worldscore-ocr"})


@app.post("/ocr")
def ocr_endpoint():
    image_file = request.files.get("image")
    body = request.get_json(silent=True) or {}
    image_base64 = body.get("image_base64")

    if image_file is None and not image_base64:
        return jsonify({"error": "Provide an image file via multipart/form-data field 'image' or JSON field 'image_base64'."}), 400

    tmp_path = None

    try:
        tmp_path = _create_temp_image_file(image_file=image_file, image_base64=image_base64)

        result = ocr_scorecard(tmp_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"OCR request failed: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/places/autocomplete", methods=["OPTIONS"])
def places_autocomplete_options():
    return ("", 204)


@app.post("/places/autocomplete")
def places_autocomplete():
    """Proxies Google Places Autocomplete (New) so the API key stays server-side.

    Keeps the Places key in Secret Manager (exposed here as the
    GOOGLE_PLACES_API_KEY env var) instead of shipping it in the mobile app.
    Restricts results to golf courses and returns the raw Places response so the
    client can parse it directly.
    """
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return jsonify({"error": "Places lookup is not configured."}), 503

    body = request.get_json(silent=True) or {}
    user_input = (body.get("input") or "").strip()
    if len(user_input) < 2:
        return jsonify({"suggestions": []})

    payload = json.dumps(
        {"input": user_input, "includedPrimaryTypes": ["golf_course"]}
    ).encode("utf-8")

    proxied = urllib.request.Request(
        PLACES_AUTOCOMPLETE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(proxied, timeout=10) as resp:
            return (resp.read(), resp.status, {"Content-Type": "application/json"})
    except urllib.error.HTTPError as exc:
        return jsonify({"error": f"Places request failed: {exc.code}"}), 502
    except Exception as exc:
        return jsonify({"error": f"Places request failed: {exc}"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
