import cv2
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

selected_y = None

PREVIEW_W = 600  # Max width for live trackbar preview downscaling

"""
Function: Interactive interface to allow user to select waterline on image input for reflection effect
- Why: To allow user to set desired waterline for reflection effect
"""
def pick_reflection_line(image):
    global selected_y, window_closed
    selected_y = None
    window_closed = False

    #Display scaling to maintain responsiveness
    MAX_W, MAX_H = 1200, 750
    h, w = image.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0) #Locks aspect ratio
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    WIN = "Step 1 — Click waterline, press ENTER to confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    #Capture user clicks to set waterline
    def mouse_callback(event, x, y, flags, param):
        global selected_y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_y = y

    cv2.setMouseCallback(WIN, mouse_callback)

    #Color palette — matches Phase 2 UI
    ACCENT  = ( 55, 180, 255)  # cyan
    GREEN   = ( 80, 200, 120)
    SUB     = (110, 110, 110)
    WHITE   = (255, 255, 255)
    BG      = ( 20,  20,  20)
    PANEL_H = 52

    while True:
        # Build canvas: dark background with image composited on top
        try:
            wr    = cv2.getWindowImageRect(WIN)
            cur_w = max(wr[2], disp_w)
            cur_h = max(wr[3], disp_h)
        except Exception:
            cur_w, cur_h = disp_w, disp_h

        canvas = np.full((cur_h, cur_w, 3), BG, dtype=np.uint8)

        # Centre image on canvas
        img_w = min(disp_w, cur_w)
        img_h = min(disp_h, cur_h - PANEL_H)
        if img_h > 0 and img_w > 0:
            # Scale to fit available space below panel
            fs  = min(cur_w / disp_w, (cur_h - PANEL_H) / disp_h)
            sw  = max(1, int(disp_w * fs))
            sh  = max(1, int(disp_h * fs))
            img_scaled = cv2.resize(display, (sw, sh), interpolation=cv2.INTER_LINEAR)
            ox = (cur_w - sw) // 2
            oy = PANEL_H + (cur_h - PANEL_H - sh) // 2
            canvas[oy:oy + sh, ox:ox + sw] = img_scaled

            if selected_y is not None:
                # Map display-space selected_y into canvas space
                canvas_y = oy + int(selected_y * fs)
                if 0 <= canvas_y < cur_h:
                    # Full-width guide line
                    cv2.line(canvas, (0, canvas_y), (cur_w, canvas_y), ACCENT, 1, cv2.LINE_AA)
                    # Tick marks every 80px
                    for x in range(0, cur_w, 80):
                        cv2.line(canvas, (x, canvas_y - 5), (x, canvas_y + 5), ACCENT, 1, cv2.LINE_AA)

        # Header panel — solid dark bar at top
        cv2.rectangle(canvas, (0, 0), (cur_w, PANEL_H), (30, 30, 30), -1)
        cv2.line(canvas, (0, PANEL_H), (cur_w, PANEL_H), (50, 50, 50), 1)

        # Step label
        cv2.putText(canvas, "Step 1", (16, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, SUB, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Set Waterline", (16, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

        # Right side of header: status or hint
        if selected_y is not None:
            real_y = int(selected_y / scale)
            status = f"y = {real_y}px"
            (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.putText(canvas, status, (cur_w - tw - 16, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
            hint = "ENTER confirm   ESC cancel"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
            cv2.putText(canvas, hint, (cur_w - hw - 16, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, GREEN, 1, cv2.LINE_AA)
        else:
            hint = "Click to set waterline   ESC cancel"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, hint, (cur_w - hw - 16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, SUB, 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(20) & 0xFF

        # X button: getWindowProperty returns -1 when window is gone
        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1

        if prop < 1: #Window closed with X-button
            cv2.destroyAllWindows()
            print("Window closed — exiting.")
            sys.exit(0)

        if key in (13, 10): # "ENTER" TO CONFIRM SELECTION
            if selected_y is not None:
                break
        elif key == 27: #"ESC" TO EXIT INTERFACE
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)

    cv2.destroyAllWindows()

    # Convert display-space y back to original image space
    real_y = int(selected_y / scale)
    return real_y


"""
PROCESSING STEPS TO CREATE TRUE WATER REFLECTION EFFECT:
1. Vertical Compression: To simulate depth and perspective by making reflection look shorter
2. Perspective Warp: To create a more realistic reflection by simulating how it would narrow towards the bottom
3. Ripple Distortion: To simulate the natural ripples and waves on a water surface, adding realism to the reflection
4. Blur: To soften the reflection and enhance realism of water reflection
5. Darkening and Vertical Fading: To simulate depth and light attenuation
6. Composite: To seamlessly combine the original image with the processed reflection, creating a cohesive final image
"""

"""
Function: Apply vertical compression to reflected image
- Why: To simulate depth and perspective by making reflection look shorter
"""
def apply_vertical_compression(reflected, w, vertical_compression):
    src_h = reflected.shape[0]
    compressed_h = max(1, int(src_h * vertical_compression))
    reflected = cv2.resize(reflected, (w, compressed_h), interpolation=cv2.INTER_LINEAR)
    return reflected, compressed_h

"""
Function: Apply perspective warp to reflected image
- Why: To create a more realistic reflection by simulating how it would narrow towards the bottom
"""
def apply_perspective_warp(reflected, w, compressed_h, perspective_shrink):
    inset = int(w * perspective_shrink / 2)
    src = np.float32([[0, 0], [w-1, 0], [0, compressed_h-1], [w-1, compressed_h-1]])
    dst = np.float32([[0, 0], [w-1, 0], [inset, compressed_h-1], [w-1-inset, compressed_h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(reflected, M, (w, compressed_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

"""
Function: Apply ripple distortion to reflected image
- Why: To simulate the natural ripples and waves on a water surface, adding realism to the reflection
"""
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

"""
Function: Applying Gaussian blur to reflected image
- Why: To soften the reflection and enhance realism of water reflection
"""
def apply_blur(reflected, blur_size):
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_size = max(1, blur_size)
    return cv2.GaussianBlur(reflected, (blur_size, blur_size), 0)

"""
Function: Apply darkening and vertical fading to reflected image
- Why: To simulate depth and light attenuation
- Darkening makes the reflection look more natural and less vibrant than the original object
- Vertical fading creates a gradient effect that mimics how reflections typically fade out with distance on water
"""
def apply_fade(reflected, compressed_h, darken, fade_min):
    reflected = np.clip(reflected.astype(np.float32) * darken, 0, 255)
    fade = np.linspace(1.0, fade_min, compressed_h).reshape(compressed_h, 1, 1)
    return np.clip(reflected * fade, 0, 255).astype(np.uint8)

"""
Function: Compile original and reflected image into one final image with waterline blending
- Why: To seamlessly combine the original image with the processed reflection, creating a cohesive final image
- Waterline blending is used to hide the seam between original and reflected
- Vectorised alpha blend across the blend band replaces the old per-row Python loop for efficiency
"""
def composite(img, reflected, reflection_start_y, compressed_h, w):
    final_h = reflection_start_y + compressed_h
    output = np.zeros((final_h, w, 3), dtype=np.uint8)
    output[:reflection_start_y] = img[:reflection_start_y]
    output[reflection_start_y:reflection_start_y + compressed_h] = reflected

    #Vectorised waterline blending to hide seam between original and reflection
    blend_height = min(8, compressed_h)
    if reflection_start_y > 0 and blend_height > 0:
        alphas    = np.linspace(0, 1, blend_height).reshape(-1, 1, 1).astype(np.float32)
        top_row   = img[reflection_start_y - 1].astype(np.float32)
        blend_src = reflected[:blend_height].astype(np.float32)
        output[reflection_start_y:reflection_start_y + blend_height] = np.clip(
            (1 - alphas) * top_row + alphas * blend_src, 0, 255
        ).astype(np.uint8)

    return output


"""
Function: Internal helper — runs the processing pipeline on the downscaled preview image
- Why: Keeps the live trackbar preview fast by operating on a small image instead of full resolution
- Uses dirty-flag caching so only pipeline stages whose inputs changed are recomputed:
    compress/warp -> ripple -> blur -> fade
    Moving darken or fade_min only reruns fade (cheapest step)
    Moving blur_size reruns blur and fade, reuses compress/warp/ripple cache
    Moving wave params reruns ripple, blur, and fade, reuses compress/warp cache
    Moving compress or perspective invalidates all stages
"""
def _build_preview(params, base_flipped, prev_w, prev_img, prev_y, cache, last_params):
    lp = last_params

    #Dirty flag checks — determine which stages need recomputing
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
    # Fade is always cheap — always recomputed, never cached

    #Recompute from earliest dirty stage downwards
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

    #Fade always runs — too cheap to bother caching
    r = apply_fade(r.copy(), ch, params["darken"], params["fade_min"])

    return composite(prev_img, r, prev_y, ch, prev_w)


"""
Function: Phase 2 — Live trackbar parameter tuning on a downscaled preview image
- Why: Lets the user interactively adjust all 7 reflection parameters and see results instantly
- Downscaling to PREVIEW_W keeps the preview responsive regardless of source image resolution
- ENTER locks the parameters and triggers the full-resolution render
- ESC or closing the window exits the program
"""
def tune_parameters(img, reflection_start_y, defaults):
    """
    Phase 2: custom side-by-side UI.
    Left panel  = drawn sliders (mouse-interactive).
    Right panel = live preview, scales dynamically with window resize.
    ENTER -> locked params for full-res render.  ESC/X -> exit.
    """
    h, w = img.shape[:2]

    # Downscale preview source
    prev_scale   = min(PREVIEW_W / w, 1.0)
    prev_w       = int(w * prev_scale)
    prev_h       = int(h * prev_scale)
    prev_img     = cv2.resize(img, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
    prev_y       = max(1, int(reflection_start_y * prev_scale))
    base_flipped = cv2.flip(prev_img[:prev_y], 0)

    # Colour palette
    BG     = ( 20,  20,  20)
    PANEL  = ( 30,  30,  30)
    ACCENT = ( 55, 180, 255)
    SUB    = (110, 110, 110)
    TRACK  = ( 55,  55,  55)
    WHITE  = (255, 255, 255)
    GREEN  = ( 80, 200, 120)

    # Layout constants
    PANEL_W  = 290
    PAD      = 20
    HEADER_H = 56
    ITEM_H   = 56
    FOOTER_H = 72

    # Slider definitions: (display label, param key, int_min, int_max, int_default, divisor)
    BARS = [
        ("Compress",    "vertical_compression", 50, 100, int(defaults["vertical_compression"] * 100), 100),
        ("Perspective", "perspective_shrink",    0,  30, int(defaults["perspective_shrink"]   * 100), 100),
        ("Wave Amp",    "wave_amp",              0,  30, int(defaults["wave_amp"]),                     1),
        ("Wave Freq",   "wave_freq",             1,  20, int(defaults["wave_freq"]            * 100), 100),
        ("Blur",        "blur_size",             1,  21, max(1, defaults["blur_size"]),                 1),
        ("Darken",      "darken",               30, 100, int(defaults["darken"]              * 100), 100),
        ("Fade Min",    "fade_min",              0,  50, int(defaults["fade_min"]            * 100), 100),
    ]

    values   = {key: default for _, key, _, _, default, _ in BARS}
    dragging = [None]

    # Slider geometry helpers
    def track_x_range():
        return PAD + 4, PANEL_W - PAD - 4

    def track_y(idx):
        return HEADER_H + idx * ITEM_H + 38

    def val_to_x(val, lo, hi):
        x1, x2 = track_x_range()
        return int(x1 + (val - lo) / max(hi - lo, 1) * (x2 - x1))

    def x_to_val(x, lo, hi):
        x1, x2 = track_x_range()
        t = float(np.clip((x - x1) / max(x2 - x1, 1), 0.0, 1.0))
        return int(round(lo + t * (hi - lo)))

    # Mouse callback for dragging sliders
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (_, key, lo, hi, _, _) in enumerate(BARS):
                ty = track_y(i)
                hx = val_to_x(values[key], lo, hi)
                x1, x2 = track_x_range()
                if (abs(x - hx) < 14 and abs(y - ty) < 14) or                    (x1 <= x <= x2 and abs(y - ty) < 10):
                    dragging[0] = i
                    values[key] = x_to_val(x, lo, hi)
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging[0] is not None:
                i = dragging[0]
                _, key, lo, hi, _, _ = BARS[i]
                values[key] = x_to_val(x, lo, hi)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging[0] = None

    # Draw left control panel onto a numpy array
    def draw_panel(canvas_h):
        panel = np.full((canvas_h, PANEL_W, 3), PANEL, dtype=np.uint8)

        # Header
        cv2.putText(panel, "Parameters", (PAD, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, WHITE, 1, cv2.LINE_AA)
        cv2.line(panel, (PAD, 46), (PANEL_W - PAD, 46), (50, 50, 50), 1)

        for i, (label, key, lo, hi, _, div) in enumerate(BARS):
            ty  = track_y(i)
            ly  = HEADER_H + i * ITEM_H + 20
            val = values[key]
            hx  = val_to_x(val, lo, hi)
            x1, x2 = track_x_range()

            # Label left, value right
            cv2.putText(panel, label, (PAD, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, SUB, 1, cv2.LINE_AA)
            disp = f"{val / div:.2f}" if div > 1 else str(val)
            (tw, _), _ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(panel, disp, (PANEL_W - PAD - tw - 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, ACCENT, 1, cv2.LINE_AA)

            # Track: grey bg then filled accent up to handle
            cv2.line(panel, (x1, ty), (x2, ty), TRACK, 3, cv2.LINE_AA)
            if hx > x1:
                cv2.line(panel, (x1, ty), (hx, ty), ACCENT, 3, cv2.LINE_AA)

            # Handle circle
            active = dragging[0] == i
            cv2.circle(panel, (hx, ty), 7, WHITE if active else ACCENT, -1, cv2.LINE_AA)
            cv2.circle(panel, (hx, ty), 7, (160, 160, 160) if active else ACCENT, 1, cv2.LINE_AA)

        # Footer hints
        fy = HEADER_H + len(BARS) * ITEM_H + 16
        cv2.line(panel, (PAD, fy), (PANEL_W - PAD, fy), (50, 50, 50), 1)
        cv2.putText(panel, "ENTER  render & save", (PAD, fy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, GREEN, 1, cv2.LINE_AA)
        cv2.putText(panel, "ESC    cancel", (PAD, fy + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, SUB, 1, cv2.LINE_AA)
        return panel

    def read_params():
        out = {key: values[key] / div for _, key, _, _, _, div in BARS}
        out["blur_size"] = int(values["blur_size"])
        return out

    # Window setup
    panel_h = HEADER_H + len(BARS) * ITEM_H + FOOTER_H
    win_h   = max(panel_h, prev_h)
    win_w   = PANEL_W + prev_w + 40
    WIN = "Step 2 — Tune parameters | ENTER = render & save | ESC = cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w, win_h)
    cv2.setMouseCallback(WIN, mouse_cb)

    cache       = {}
    last_params = {}
    preview     = None

    while True:
        params = read_params()
        if params != last_params:
            preview = _build_preview(params, base_flipped, prev_w, prev_img, prev_y,
                                     cache, last_params)
            last_params = params.copy()

        # Read actual window size for dynamic scaling
        try:
            wr    = cv2.getWindowImageRect(WIN)
            cur_w = max(wr[2], PANEL_W + 10)
            cur_h = max(wr[3], 200)
        except Exception:
            cur_w, cur_h = win_w, win_h

        # Build full canvas
        canvas = np.full((cur_h, cur_w, 3), BG, dtype=np.uint8)

        # Left panel
        panel = draw_panel(cur_h)
        ph    = min(panel.shape[0], cur_h)
        canvas[:ph, :PANEL_W] = panel[:ph]

        # Vertical divider line
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, cur_h), (45, 45, 45), 1)

        # Right side: preview centred in available space
        if preview is not None:
            avail_w = cur_w - PANEL_W - 1
            avail_h = cur_h
            if avail_w > 1 and avail_h > 1:
                ph2, pw2 = preview.shape[:2]
                fs  = min(avail_w / pw2, avail_h / ph2)
                dw  = max(1, int(pw2 * fs))
                dh  = max(1, int(ph2 * fs))
                scaled = cv2.resize(preview, (dw, dh), interpolation=cv2.INTER_LINEAR)
                ox  = PANEL_W + 1 + (avail_w - dw) // 2
                oy  = (avail_h - dh) // 2
                canvas[oy:oy + dh, ox:ox + dw] = scaled

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(20) & 0xFF

        try:
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            prop = -1

        if prop < 1:
            cv2.destroyAllWindows()
            print("Window closed — exiting.")
            sys.exit(0)

        if key in (13, 10): #ENTER — lock params and trigger full-res render
            cv2.destroyAllWindows()
            return params
        elif key == 27: #ESC TO EXIT
            cv2.destroyAllWindows()
            print("Cancelled.")
            sys.exit(0)



"""
Function: Phase 3 — Render the final composite at full resolution and save to disk
- Why: Preview runs on a downscaled image for speed; this step produces the actual output
- Only runs once after the user confirms parameters in Phase 2
"""
def render_full_res(img, reflection_start_y, params, output_path, jpeg_quality=95):
    h, w = img.shape[:2]
    print("Rendering full resolution...")

    #Extract region above waterline and reflect vertically to create base reflection
    source_region = img[:reflection_start_y, :, :]
    reflected = cv2.flip(source_region, 0)

    #Check for valid waterline to ensure enough processing space
    if reflected.shape[0] == 0:
        raise ValueError("Reflection line is too close to the top of the image.")

    #Apply full processing pipeline at full resolution using locked parameters
    reflected, compressed_h = apply_vertical_compression(reflected, w, params["vertical_compression"])
    reflected = apply_perspective_warp(reflected, w, compressed_h, params["perspective_shrink"])
    reflected = apply_ripple(reflected, w, compressed_h, params["wave_amp"], params["wave_freq"])
    reflected = apply_blur(reflected, params["blur_size"])
    reflected = apply_fade(reflected, compressed_h, params["darken"], params["fade_min"])

    #Create final composite image and save to results directory
    output = composite(img, reflected, reflection_start_y, compressed_h, w)
    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved to {output_path}")


"""
Function: Main function to create interactive lake reflection effect on an image
- Orchestrates the three-phase pipeline:
    Phase 1: User selects waterline interactively
    Phase 2: User tunes parameters with live trackbar preview on downscaled image
    Phase 3: Full-resolution render and save using confirmed parameters

Parameters:
- perspective_shrink: How much the reflection narrows towards the bottom to enhance perspective
- vertical_compression: Compresses the reflection vertically to simulate depth and prevent it from looking too tall
- wave_amp: Amplitude of the ripple distortion
- wave_freq: Frequency of the ripple distortion
- blur_size: Blur applied to the reflection
- darken: Level of darkening applied to the reflection
- fade_min: Minimum fade value for the vertical fade effect
- jpeg_quality: Quality of the saved JPEG image (0-100)
"""
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
    #Load image and validate
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    #Phase 1: Get waterline from user via interactive interface
    reflection_start_y = pick_reflection_line(img)
    reflection_start_y = max(1, min(reflection_start_y, h - 1))

    #Phase 2: Tune parameters interactively on a downscaled preview
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

    #Phase 3: Render at full resolution with confirmed parameters and save
    render_full_res(img, reflection_start_y, params, output_path, jpeg_quality)


"""
MAIN:
- Directory setup for input images and output results
- Image selection via user input with multi-format support
- Auto-timestamped output filename to avoid overwriting previous results
"""
if __name__ == "__main__":
    #Directory setup for input images and output results
    IMAGES_DIR  = Path("images")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    #Support any image format to allow flexibility for user input
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    #Image selection via user input
    image_input = input("Enter the name of your image (without extension): ").strip()

    #Default to ferrari.jpg if no input was provided
    if image_input == "":
        image_path = IMAGES_DIR / "ferrari.jpg"
    else:
        #Search for image with any supported format extension
        image_path = None
        for ext in supported_formats:
            candidate = IMAGES_DIR / f"{image_input}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        #Valid image checking
        if image_path is None:
            print(f"Image not found: {image_input} with supported formats {supported_formats}")
            print(f"Put your images in the '{IMAGES_DIR}/' folder and specify the name without extension.")
            sys.exit(1)

    #Valid path checking
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print(f"Put your images in the '{IMAGES_DIR}/' folder.")
        sys.exit(1)

    #Output filename with timestamp to avoid overwriting previous results
    stem = image_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{stem}_reflection_{timestamp}.jpg"

    #Run the full three-phase interactive pipeline
    create_interactive_lake_reflection(
        image_path=str(image_path),
        output_path=str(output_path),
    )