import cv2
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

selected_y = None

PREVIEW_W = 600  # max width for live trackbar preview

# ── Processing steps ──────────────────────────────────────────────────────────

def apply_vertical_compression(reflected, w, vertical_compression):
    src_h = reflected.shape[0]
    compressed_h = max(1, int(src_h * vertical_compression))
    return cv2.resize(reflected, (w, compressed_h), interpolation=cv2.INTER_LINEAR), compressed_h


def apply_perspective_warp(reflected, w, compressed_h, perspective_shrink):
    inset = int(w * perspective_shrink / 2)
    src = np.float32([[0, 0], [w-1, 0], [0, compressed_h-1], [w-1, compressed_h-1]])
    dst = np.float32([[0, 0], [w-1, 0], [inset, compressed_h-1], [w-1-inset, compressed_h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(reflected, M, (w, compressed_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def apply_ripple(reflected, w, compressed_h, wave_amp, wave_freq):
    ys = np.arange(compressed_h, dtype=np.float32)
    local_amp = wave_amp * (0.25 + 0.75 * (ys / max(compressed_h - 1, 1)))
    shifts = (local_amp * np.sin(2 * np.pi * wave_freq * ys)).reshape(-1, 1)
    xs = np.tile(np.arange(w, dtype=np.float32), (compressed_h, 1))
    map_x = np.clip(xs + shifts, 0, w - 1)
    map_y = np.tile(ys.reshape(-1, 1), (1, w))
    return cv2.remap(reflected, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def apply_blur(reflected, blur_size):
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_size = max(1, blur_size)
    return cv2.GaussianBlur(reflected, (blur_size, blur_size), 0)


def apply_fade(reflected, compressed_h, darken, fade_min):
    r = np.clip(reflected.astype(np.float32) * darken, 0, 255)
    fade = np.linspace(1.0, fade_min, compressed_h).reshape(compressed_h, 1, 1)
    return np.clip(r * fade, 0, 255).astype(np.uint8)


def composite(img, reflected, reflection_start_y, compressed_h, w):
    final_h = reflection_start_y + compressed_h
    output = np.zeros((final_h, w, 3), dtype=np.uint8)
    output[:reflection_start_y] = img[:reflection_start_y]
    output[reflection_start_y:reflection_start_y + compressed_h] = reflected

    # Vectorised waterline blend (replaces the old Python loop)
    blend_h = min(8, compressed_h)
    if reflection_start_y > 0 and blend_h > 0:
        alphas   = np.linspace(0, 1, blend_h).reshape(-1, 1, 1).astype(np.float32)
        top_row  = img[reflection_start_y - 1].astype(np.float32)
        blend_src = reflected[:blend_h].astype(np.float32)
        output[reflection_start_y:reflection_start_y + blend_h] = np.clip(
            (1 - alphas) * top_row + alphas * blend_src, 0, 255
        ).astype(np.uint8)

    return output


# ── Phase 1: waterline picker ─────────────────────────────────────────────────

def pick_reflection_line(image):
    global selected_y
    selected_y = None

    MAX_W, MAX_H = 1200, 750
    h, w = image.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    WIN = "Step 1 — Click waterline, press ENTER to confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def mouse_callback(event, x, y, flags, param):
        global selected_y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_y = y

    cv2.setMouseCallback(WIN, mouse_callback)

    GREEN, YELLOW, SHADOW = (72, 199, 142), (55, 210, 255), (0, 0, 0)

    while True:
        frame = display.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, 48), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if selected_y is not None:
            cv2.line(frame, (0, selected_y), (disp_w, selected_y), GREEN, 2, cv2.LINE_AA)
            for x in range(0, disp_w, 80):
                cv2.line(frame, (x, selected_y - 5), (x, selected_y + 5), GREEN, 1, cv2.LINE_AA)
            real_y = int(selected_y / scale)
            label = f"  Waterline  y={real_y}px  |  ENTER to confirm  |  ESC to cancel"
            cv2.putText(frame, label, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, GREEN,  1, cv2.LINE_AA)
        else:
            hint = "  Click anywhere to set the waterline"
            cv2.putText(frame, hint, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW,  3, cv2.LINE_AA)
            cv2.putText(frame, hint, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, YELLOW,  1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1

        if prop < 1:
            cv2.destroyAllWindows()
            print("Window closed — exiting.")
            sys.exit(0)

        if key in (13, 10) and selected_y is not None:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)

    cv2.destroyAllWindows()
    return int(selected_y / scale)


# ── Phase 2: live trackbar tuning on downscaled preview ───────────────────────

def _build_preview(params, base_flipped, prev_w, prev_img, prev_y, cache, last_params):
    """
    Run the pipeline on the downscaled flipped region.
    Uses dirty-flag caching: only recomputes steps whose inputs changed.
    Returns the composited preview frame.
    """
    lp = last_params

    # Which pipeline stages are dirty?
    compress_dirty = (
        "warped" not in cache
        or params["vertical_compression"] != lp.get("vertical_compression")
        or params["perspective_shrink"]   != lp.get("perspective_shrink")
    )
    ripple_dirty = compress_dirty or (
        params["wave_amp"]  != lp.get("wave_amp")
        or params["wave_freq"] != lp.get("wave_freq")
    )
    blur_dirty = ripple_dirty or params["blur_size"] != lp.get("blur_size")
    # fade is always cheap — always redo

    # Recompute from the earliest dirty stage
    if compress_dirty:
        r, ch = apply_vertical_compression(base_flipped, prev_w, params["vertical_compression"])
        r = apply_perspective_warp(r, prev_w, ch, params["perspective_shrink"])
        cache["warped"] = (r.copy(), ch)

    r, ch = cache["warped"]

    if ripple_dirty:
        cache["rippled"] = apply_ripple(r, prev_w, ch, params["wave_amp"], params["wave_freq"])

    r = cache["rippled"]

    if blur_dirty:
        cache["blurred"] = apply_blur(r.copy(), params["blur_size"])

    r = cache["blurred"]

    # Fade: always fast, skip caching
    r = apply_fade(r.copy(), ch, params["darken"], params["fade_min"])

    return composite(prev_img, r, prev_y, ch, prev_w)


def tune_parameters(img, reflection_start_y, defaults):
    """
    Phase 2: live trackbar window on a downscaled preview.
    ENTER  → returns the final parameter dict for full-res rendering.
    ESC/X  → exits the program.
    """
    h, w = img.shape[:2]

    # Downscale image and waterline for preview
    prev_scale = min(PREVIEW_W / w, 1.0)
    prev_w = int(w * prev_scale)
    prev_h = int(h * prev_scale)
    prev_img = cv2.resize(img, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
    prev_y   = max(1, int(reflection_start_y * prev_scale))

    # Pre-flip the preview source region (never changes)
    base_flipped = cv2.flip(prev_img[:prev_y], 0)

    WIN = "Step 2 — Tune parameters | ENTER = render & save | ESC = cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Trackbar spec: (label, min, max, default_int, divisor_to_float)
    BARS = [
        ("Vert Compress %",  50, 100, int(defaults["vertical_compression"] * 100), 100),
        ("Perspective x100",  0,  30, int(defaults["perspective_shrink"]   * 100), 100),
        ("Wave Amp",          0,  30, int(defaults["wave_amp"]),                     1),
        ("Wave Freq x100",    1,  20, int(defaults["wave_freq"]            * 100), 100),
        ("Blur Size",         1,  21, max(1, defaults["blur_size"]),                 1),
        ("Darken %",         30, 100, int(defaults["darken"]               * 100), 100),
        ("Fade Min %",        0,  50, int(defaults["fade_min"]             * 100), 100),
    ]

    for name, lo, hi, default, _ in BARS:
        cv2.createTrackbar(name, WIN, default, hi, lambda v: None)
        cv2.setTrackbarMin(name, WIN, lo)

    # Resize window AFTER trackbars are created so the image area is correct.
    # Each OpenCV trackbar is ~30px tall; we account for all 7 of them.
    TRACKBAR_H = len(BARS) * 30
    win_w = max(prev_w, 700)           # at least 700px wide so labels aren't clipped
    win_h = prev_h + TRACKBAR_H
    cv2.resizeWindow(WIN, win_w, win_h)

    def read_params():
        raw = {name: cv2.getTrackbarPos(name, WIN) for name, *_ in BARS}
        divisors = {name: div for name, _, _, _, div in BARS}
        return {
            "vertical_compression": raw["Vert Compress %"]  / divisors["Vert Compress %"],
            "perspective_shrink":   raw["Perspective x100"] / divisors["Perspective x100"],
            "wave_amp":             raw["Wave Amp"]          / divisors["Wave Amp"],
            "wave_freq":            raw["Wave Freq x100"]    / divisors["Wave Freq x100"],
            "blur_size":        int(raw["Blur Size"]),
            "darken":               raw["Darken %"]          / divisors["Darken %"],
            "fade_min":             raw["Fade Min %"]        / divisors["Fade Min %"],
        }

    cache      = {}
    last_params = {}
    preview    = None

    while True:
        params = read_params()

        # Only rebuild if something actually changed
        if params != last_params:
            preview = _build_preview(params, base_flipped, prev_w, prev_img, prev_y,
                                     cache, last_params)
            last_params = params.copy()

        if preview is not None:
            frame = preview.copy()
            fh, fw = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 40), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            hint = "  ENTER = render full-res & save   |   ESC = cancel"
            cv2.putText(frame, hint, (11, 27), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0,   0,   0),   2, cv2.LINE_AA)
            cv2.putText(frame, hint, (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (55, 210, 255),  1, cv2.LINE_AA)
            cv2.imshow(WIN, frame)

        key = cv2.waitKey(30) & 0xFF

        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1

        if prop < 1:
            cv2.destroyAllWindows()
            print("Window closed — exiting.")
            sys.exit(0)

        if key in (13, 10):   # ENTER — lock params and move to full-res render
            cv2.destroyAllWindows()
            return params
        elif key == 27:        # ESC
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)


# ── Phase 3: full-res render + save ──────────────────────────────────────────

def render_full_res(img, reflection_start_y, params, output_path, jpeg_quality=95):
    h, w = img.shape[:2]
    print("Rendering full resolution...")

    source_region = img[:reflection_start_y]
    reflected = cv2.flip(source_region, 0)

    if reflected.shape[0] == 0:
        raise ValueError("Reflection line is too close to the top of the image.")

    reflected, compressed_h = apply_vertical_compression(reflected, w, params["vertical_compression"])
    reflected = apply_perspective_warp(reflected, w, compressed_h, params["perspective_shrink"])
    reflected = apply_ripple(reflected, w, compressed_h, params["wave_amp"], params["wave_freq"])
    reflected = apply_blur(reflected, params["blur_size"])
    reflected = apply_fade(reflected, compressed_h, params["darken"], params["fade_min"])

    output = composite(img, reflected, reflection_start_y, compressed_h, w)
    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved to {output_path}")


# ── Main entry ────────────────────────────────────────────────────────────────

def create_interactive_lake_reflection(
    image_path,
    output_path="interactive_reflection.jpg",
    perspective_shrink=0.12,
    vertical_compression=0.82,
    wave_amp=5,
    wave_freq=0.06,
    blur_size=5,
    darken=0.78,
    fade_min=0.12,
    jpeg_quality=95
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    # Phase 1: pick waterline
    reflection_start_y = pick_reflection_line(img)
    reflection_start_y = max(1, min(reflection_start_y, h - 1))

    # Phase 2: tune parameters interactively on a downscaled preview
    defaults = {
        "vertical_compression": vertical_compression,
        "perspective_shrink":   perspective_shrink,
        "wave_amp":             wave_amp,
        "wave_freq":            wave_freq,
        "blur_size":            blur_size,
        "darken":               darken,
        "fade_min":             fade_min,
    }
    params = tune_parameters(img, reflection_start_y, defaults)

    # Phase 3: render at full resolution with locked params
    render_full_res(img, reflection_start_y, params, output_path, jpeg_quality)


if __name__ == "__main__":
    IMAGES_DIR  = Path("images")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    image_input = input("Enter the name of your image (without extension): ").strip()

    if image_input == "":
        image_path = IMAGES_DIR / "ferrari.jpg"
    else:
        image_path = None
        for ext in supported_formats:
            candidate = IMAGES_DIR / f"{image_input}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"Image '{image_input}' not found with supported formats: {supported_formats}")
            print(f"Put your images in the '{IMAGES_DIR}/' folder and specify the name without extension.")
            sys.exit(1)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print(f"Put your images in the '{IMAGES_DIR}/' folder.")
        sys.exit(1)

    stem        = image_path.stem
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{stem}_reflection_{timestamp}.jpg"

    create_interactive_lake_reflection(
        image_path=str(image_path),
        output_path=str(output_path),
    )