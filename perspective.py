import cv2
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

# ── Display constants ─────────────────────────────────────────────────────────
MAX_W, MAX_H = 1200, 720
GREEN  = (72, 199, 142)
YELLOW = (55, 210, 255)
SHADOW = (0, 0, 0)

selected_y = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — pick the waterline
# ─────────────────────────────────────────────────────────────────────────────
def pick_reflection_line(image):
    global selected_y
    selected_y = None

    h, w   = image.shape[:2]
    scale  = min(MAX_W / w, MAX_H / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    WIN = "Step 1 of 2 — Click the waterline, then press ENTER"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def on_mouse(event, x, y, flags, param):
        global selected_y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_y = y

    cv2.setMouseCallback(WIN, on_mouse)

    PANEL_H = 48
    while True:
        frame = display.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, PANEL_H), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if selected_y is not None:
            cv2.line(frame, (0, selected_y), (disp_w, selected_y), GREEN, 2, cv2.LINE_AA)
            for x in range(0, disp_w, 80):
                cv2.line(frame, (x, selected_y - 5), (x, selected_y + 5), GREEN, 1, cv2.LINE_AA)
            real_y = int(selected_y / scale)
            msg = f"  Waterline y={real_y}px  |  ENTER to continue  |  ESC to quit"
            cv2.putText(frame, msg, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, GREEN,  1, cv2.LINE_AA)
        else:
            msg = "  Click anywhere to set the waterline"
            cv2.putText(frame, msg, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW,  3, cv2.LINE_AA)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, YELLOW,  1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1
        if prop < 1:
            cv2.destroyAllWindows()
            print("Closed — exiting.")
            sys.exit(0)

        if key in (13, 10) and selected_y is not None:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)

    cv2.destroyWindow(WIN)
    return int(selected_y / scale)   # back to original-image coords


# ─────────────────────────────────────────────────────────────────────────────
# Core reflection renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_reflection(img, reflection_start_y,
                      wave_amp, wave_freq, blur_size,
                      darken, fade_min,
                      vertical_compression, perspective_shrink):

    h, w = img.shape[:2]
    reflection_start_y = max(1, min(reflection_start_y, h - 1))

    top_part  = img[:reflection_start_y].copy()
    reflected = cv2.flip(img[:reflection_start_y], 0)
    src_h     = reflected.shape[0]

    # Vertical compression
    compressed_h = max(1, int(src_h * vertical_compression))
    reflected = cv2.resize(reflected, (w, compressed_h), interpolation=cv2.INTER_LINEAR)

    # Perspective warp
    inset    = int(w * perspective_shrink / 2)
    src_pts  = np.float32([[0,0],[w-1,0],[0,compressed_h-1],[w-1,compressed_h-1]])
    dst_pts  = np.float32([[0,0],[w-1,0],[inset,compressed_h-1],[w-1-inset,compressed_h-1]])
    M        = cv2.getPerspectiveTransform(src_pts, dst_pts)
    reflected = cv2.warpPerspective(reflected, M, (w, compressed_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)

    # Ripple (vectorised)
    ys        = np.arange(compressed_h, dtype=np.float32)
    local_amp = wave_amp * (0.25 + 0.75 * (ys / max(compressed_h - 1, 1)))
    shifts    = (local_amp * np.sin(2 * np.pi * wave_freq * ys)).reshape(-1, 1)
    xs        = np.tile(np.arange(w, dtype=np.float32), (compressed_h, 1))
    map_x     = np.clip(xs + shifts, 0, w - 1)
    map_y     = np.tile(ys.reshape(-1, 1), (1, w))
    reflected = cv2.remap(reflected, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # Blur (must be odd)
    bs = max(1, int(blur_size)) | 1
    reflected = cv2.GaussianBlur(reflected, (bs, bs), 0)

    # Darken + fade
    ref_f  = reflected.astype(np.float32) * darken
    fade   = np.linspace(1.0, fade_min, compressed_h).reshape(compressed_h, 1, 1)
    ref_f *= fade
    reflected = np.clip(ref_f, 0, 255).astype(np.uint8)

    # Composite
    final_h = reflection_start_y + compressed_h
    out     = np.zeros((final_h, w, 3), dtype=np.uint8)
    out[:reflection_start_y] = top_part
    out[reflection_start_y:reflection_start_y + compressed_h] = reflected

    # Blend band at waterline
    for i in range(min(8, compressed_h)):
        alpha = i / 8
        row   = reflection_start_y + i
        if row < final_h and reflection_start_y > 0:
            out[row] = np.clip(
                (1 - alpha) * img[reflection_start_y - 1].astype(np.float32) +
                alpha        * reflected[i].astype(np.float32),
                0, 255
            ).astype(np.uint8)

    cv2.line(out, (0, reflection_start_y), (w, reflection_start_y), (35, 35, 35), 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — live trackbar tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_and_confirm(img, reflection_start_y):
    h, w  = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    prev_w = int(w * scale)
    prev_h = int(h * scale)

    WIN = "Step 2 of 2 — Tune the reflection  |  ENTER to save  |  ESC to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, prev_w, prev_h)

    # ── Trackbars ─────────────────────────────────────────────────────────────
    # Stored as integers; divide by divisor to get the real float value.
    #  (label, default_int, max_int, divisor)
    BARS = [
        ("Wave Amplitude  (0-30 px)",   5,   30,   1  ),
        ("Wave Frequency x100",          6,   30,   100),
        ("Blur Size       (1-21)",       5,   21,   1  ),
        ("Darken %",                    78,  100,   100),
        ("Fade to %",                   12,  100,   100),
        ("Compression %",               82,  100,   100),
        ("Perspective %",               12,   40,   100),
    ]

    for label, default, maximum, _ in BARS:
        cv2.createTrackbar(label, WIN, default, maximum, lambda v: None)

    def read_params():
        v = [cv2.getTrackbarPos(label, WIN) for label, *_ in BARS]
        return (
            max(0,     v[0]),           # wave_amp
            max(0.001, v[1] / 100),     # wave_freq
            max(1,     v[2]),           # blur_size
            max(0.05,  v[3] / 100),     # darken
            max(0.01,  v[4] / 100),     # fade_min
            max(0.05,  v[5] / 100),     # vertical_compression
            max(0.0,   v[6] / 100),     # perspective_shrink
        )

    last_params = None

    while True:
        params = read_params()

        if params != last_params:
            amp, freq, blur, dark, fade, comp, persp = params
            result  = render_reflection(img, reflection_start_y,
                                        amp, freq, blur, dark, fade, comp, persp)
            preview = cv2.resize(result, (prev_w, prev_h), interpolation=cv2.INTER_AREA)

            # Status bar overlay
            overlay = preview.copy()
            cv2.rectangle(overlay, (0, 0), (prev_w, 44), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.75, preview, 0.25, 0, preview)
            msg = (f"  amp={amp}  freq={freq:.3f}  blur={blur}  "
                   f"dark={dark:.0%}  fade={fade:.0%}  "
                   f"comp={comp:.0%}  persp={persp:.0%}"
                   f"  |  ENTER to save  |  ESC to quit")
            cv2.putText(preview, msg, (11, 29), cv2.FONT_HERSHEY_DUPLEX, 0.45, SHADOW, 3, cv2.LINE_AA)
            cv2.putText(preview, msg, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.45, GREEN,  1, cv2.LINE_AA)

            cv2.imshow(WIN, preview)
            last_params = params

        key = cv2.waitKey(30) & 0xFF

        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1
        if prop < 1:
            cv2.destroyAllWindows()
            print("Closed — exiting.")
            sys.exit(0)

        if key in (13, 10):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)

    cv2.destroyAllWindows()

    amp, freq, blur, dark, fade, comp, persp = params
    return dict(wave_amp=amp, wave_freq=freq, blur_size=blur,
                darken=dark, fade_min=fade,
                vertical_compression=comp, perspective_shrink=persp)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_interactive_lake_reflection(image_path, output_path, jpeg_quality=95):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    reflection_start_y = pick_reflection_line(img)
    params             = tune_and_confirm(img, reflection_start_y)

    result = render_reflection(img, reflection_start_y, **params)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved → {output_path}")
    print("Final params:", params)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IMAGES_DIR  = Path("images")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    image_path = IMAGES_DIR / (sys.argv[1] if len(sys.argv) > 1 else "ferrari.jpg")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print(f"Put your images in the '{IMAGES_DIR}/' folder.")
        sys.exit(1)

    stem        = image_path.stem
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{stem}_reflection_{timestamp}.jpg"

    create_interactive_lake_reflection(image_path, output_path)
