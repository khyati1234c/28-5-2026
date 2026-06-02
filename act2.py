import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = OSgetenv("HF API KEY")
if not HF_API_KEY:
    print("HF_API_KEY NOT FOUND PLEASE ADD IT TO YOUR .env  file")
    print("HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxx")
    exit(1)

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADER = {
    "Authorization": f"Bearer {HF_API_KEY}"
    "Content-type" : "application/json
}

MODEL = [
   "Qwen/Qwen2.5-VL-72B-Instruct", 
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    "google/gemma-3-12b-it", 
]

ALLOWED = ["jpg","jpeg",,"png","webp","bmp"]

def to_data_url(img_bytes: bytes, path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode()


def extract_error(r: requests.Response) -> str:
    """Pull the most useful error message from an API response."""
    try:
        j = r.json()
        return j.get("error", {}).get("message") or str(j)
    except Exception:
        return (r.text or "").strip() or r.reason or "Unknown error"


def print_box(title: str, lines: list, icon: str = ""):
    """Print a pretty unicode box."""
    width = max(30, len(title) + 4, *(len(x) for x in lines))
    print("\n" + "┏" + "━" * (width + 2) + "┓")
    print(f"┃ {icon + ' ' if icon else ''}{title.ljust(width - (2 if icon else 0))} ┃")
    print("┣" + "━" * (width + 2) + "┫")
    for line in lines:
        print(f"┃ {line.ljust(width)} ┃")
    print("┗" + "━" * (width + 2) + "┛\n")


def load_image(path: str) -> bytes | None:
    """Validate and load an image file. Returns bytes or None on failure."""
    if not os.path.isfile(path):
        print_box("File Error", [f"File not found: {path}"], "❌")
        return None
    if os.path.splitext(path)[1].lower() not in ALLOWED:
        print_box("File Error", [f"Unsupported format. Use: {', '.join(ALLOWED)}"], "❌")
        return None
    if os.path.getsize(path) / (1024 * 1024) > 10:
        print_box("File Error", ["File too large (>10MB). Please resize."], "❌")
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        print_box("File Error", [f"Could not read file: {e}"], "❌")
        return None


# ── Core ────────────────────────────────────────────────────────
def caption_image(image_path: str, prompt: str = "Give a short, descriptive caption for this image.") -> str | None:
    """
    Send image to VLM and return caption string, or None on total failure.
    Tries each model in MODELS order until one succeeds.
    """
    img_bytes = load_image(image_path)
    if img_bytes is None:
        return None

    payload_base = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(img_bytes, image_path)}},
            ],
        }],
        "max_tokens": 120,
        "temperature": 0.2,
    }

    last_error = None

    for model in MODELS:
        print(f"  → Trying: {model}")
        try:
            r = requests.post(
                API_URL,
                headers=HEADERS,
                json={**payload_base, "model": model},
                timeout=120,
            )
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            print(f"  ✗ {last_error}")
            continue
        except requests.exceptions.Timeout:
            last_error = "Request timed out."
            print(f"  ✗ {last_error}")
            continue

        if r.status_code != 200:
            last_error = extract_error(r)
            print(f"  ✗ HTTP {r.status_code}: {last_error}")
            continue

        try:
            data    = r.json()
            caption = (data.get("choices", [{}])[0]
                          .get("message", {})
                          .get("content") or "").strip()
        except Exception as e:
            last_error = f"Could not parse response: {e}"
            print(f"  ✗ {last_error}")
            continue

        if caption:
            print(f"  ✅ Success with {model}")
            return caption

        last_error = "Empty caption returned."
        print(f"  ✗ {last_error}")

    print_box("Caption Failed", [
        f"Image : {image_path}",
        f"Error : {last_error or 'All models failed'}",
    ], "⚠️")
    return None


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("     AI Image Captioning  |  Vision-Language Model")
    print("=" * 55)

    while True:
        image_path = input("\n  Image path (or 'quit'): ").strip().strip('"').strip("'")
        if image_path.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not image_path:
            image_path = "test.jpg"

        custom = input("💬 Custom prompt? (Enter to use default): ").strip()
        prompt = custom if custom else "Give a short, descriptive caption for this image."

        print("\n🔍 Generating caption...")
        caption = caption_image(image_path, prompt)

        if caption:
            print_box("Caption Result", [
                f"Image  : {os.path.basename(image_path)}",
                f"Prompt : {prompt[:60]}{'...' if len(prompt) > 60 else ''}",
                "",
                " Caption:",
                f"   {caption}",
            ], "")


if __name__ == "__main__":
    main()

