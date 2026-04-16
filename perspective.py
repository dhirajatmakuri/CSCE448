import cv2
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path

selected_y = None
window_closed = False


def pick_reflection_line(image):
    global selected_y, window_closed
    selected_y = None
    window_closed = False

    # ── Scale down for display so it fits on screen ──────────────────────────
    MAX_W, MAX_H = 1200, 750
    h, w = image.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)          # never upscale
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    WIN = "Lake Reflection — Click waterline, press ENTER to confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def mouse_callback(event, x, y, flags, param):
        global selected_y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_y = y           # in display coords

    cv2.setMouseCallback(WIN, mouse_callback)

    # ── Colour palette ────────────────────────────────────────────────────────
    GREEN   = (72, 199, 142)
    YELLOW  = (55, 210, 255)
    SHADOW  = (0, 0, 0)
    PANEL_H = 48

    while True:
        frame = display.copy()

        # semi-transparent top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, PANEL_H), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if selected_y is not None:
            # horizontal guide line (display coords)
            cv2.line(frame, (0, selected_y), (disp_w, selected_y), GREEN, 2, cv2.LINE_AA)

            # tick marks every 80 px
            for x in range(0, disp_w, 80):
                cv2.line(frame, (x, selected_y - 5), (x, selected_y + 5), GREEN, 1, cv2.LINE_AA)

            # label
            real_y = int(selected_y / scale)          # map back to original coords
            label = f"  Waterline  y={real_y}px  |  ENTER to confirm  |  ESC to cancel"
            # shadow
            cv2.putText(frame, label, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, GREEN,  1, cv2.LINE_AA)
        else:
            hint = "  Click anywhere to set the waterline"
            cv2.putText(frame, hint, (11, 31), cv2.FONT_HERSHEY_DUPLEX, 0.6, SHADOW,  3, cv2.LINE_AA)
            cv2.putText(frame, hint, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, YELLOW,  1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        # ── X button: getWindowProperty returns -1 when window is gone ────────
        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1

        if prop < 1:                          # window was closed with X
            cv2.destroyAllWindows()
            print("Window closed — exiting.")
            sys.exit(0)

        if key in (13, 10):                   # ENTER
            if selected_y is not None:
                break
        elif key == 27:                       # ESC
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)

    cv2.destroyAllWindows()

    # Convert display-space y back to original image space
    real_y = int(selected_y / scale)
    return real_y


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

    reflection_start_y = pick_reflection_line(img)
    reflection_start_y = max(1, min(reflection_start_y, h - 1))

    top_part      = img[:reflection_start_y, :, :]
    source_region = img[:reflection_start_y, :, :]
    reflected     = cv2.flip(source_region, 0)

    src_h = reflected.shape[0]
    if src_h == 0:
        raise ValueError("Reflection line is too close to the top of the image.")

    # Vertical compression
    compressed_h = max(1, int(src_h * vertical_compression))
    reflected = cv2.resize(reflected, (w, compressed_h), interpolation=cv2.INTER_LINEAR)

    # Perspective warp
    inset = int(w * perspective_shrink / 2)
    src = np.float32([[0, 0], [w-1, 0], [0, compressed_h-1], [w-1, compressed_h-1]])
    dst = np.float32([[0, 0], [w-1, 0], [inset, compressed_h-1], [w-1-inset, compressed_h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    reflected = cv2.warpPerspective(reflected, M, (w, compressed_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)

    # Ripple distortion (vectorised — much faster than the pixel loop)
    ys = np.arange(compressed_h, dtype=np.float32)
    local_amp = wave_amp * (0.25 + 0.75 * (ys / max(compressed_h - 1, 1)))
    shifts    = (local_amp * np.sin(2 * np.pi * wave_freq * ys)).reshape(-1, 1)

    xs = np.tile(np.arange(w, dtype=np.float32), (compressed_h, 1))
    map_x = np.clip(xs + shifts, 0, w - 1)
    map_y = np.tile(ys.reshape(-1, 1), (1, w))

    reflected = cv2.remap(reflected, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # Blur
    if blur_size % 2 == 0:
        blur_size += 1
    reflected = cv2.GaussianBlur(reflected, (blur_size, blur_size), 0)

    # Darken + fade
    reflected = np.clip(reflected.astype(np.float32) * darken, 0, 255)
    fade = np.linspace(1.0, fade_min, compressed_h).reshape(compressed_h, 1, 1)
    reflected = np.clip(reflected * fade, 0, 255).astype(np.uint8)

    # Composite
    final_h = reflection_start_y + compressed_h
    output  = np.zeros((final_h, w, 3), dtype=np.uint8)
    output[:reflection_start_y] = top_part
    output[reflection_start_y:reflection_start_y + compressed_h] = reflected

    # Waterline blend band (~8 px)
    blend_height = 8
    for i in range(min(blend_height, compressed_h)):
        alpha = i / blend_height
        row   = reflection_start_y + i
        if row < final_h and (reflection_start_y - 1) >= 0:
            output[row] = np.clip(
                (1 - alpha) * img[reflection_start_y - 1].astype(np.float32) +
                alpha        * reflected[i].astype(np.float32),
                0, 255
            ).astype(np.uint8)

    # Subtle waterline
    cv2.line(output, (0, reflection_start_y), (w, reflection_start_y), (35, 35, 35), 1)

    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    # ── Folder setup ─────────────────────────────────────────────────────────
    IMAGES_DIR  = Path("images")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Pick an image: pass filename as CLI arg, or set a default ────────────
    # Usage:  python perspective.py ferrari.jpg
    #         python perspective.py skyline.jpg
    if len(sys.argv) > 1:
        image_path = IMAGES_DIR / sys.argv[1]
    else:
        image_path = IMAGES_DIR / "ferrari.jpg"   # ← change default here

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print(f"Put your images in the '{IMAGES_DIR}/' folder.")
        sys.exit(1)

    # ── Auto-generate output name: <stem>_reflection_<YYYYMMDD_HHMMSS>.jpg ──
    stem        = image_path.stem
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{stem}_reflection_{timestamp}.jpg"

    create_interactive_lake_reflection(
        image_path=str(image_path),
        output_path=str(output_path),
        perspective_shrink=0.12,
        vertical_compression=0.82,
        wave_amp=5,
        wave_freq=0.06,
        blur_size=5,
        darken=0.78,
        fade_min=0.12
    )