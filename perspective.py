import cv2
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

selected_y = None
window_closed = False

"""
Function: To allow the user to interactively select the waterline for the reflection effect.
- Displays the image in a resizable window
- Sets waterline by clicking on image

Parameters:
- image: Input image to add water reflection to
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

    WIN = "Lake Reflection — Click waterline, press ENTER to confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    #Capture user clicks to set waterline
    def mouse_callback(event, x, y, flags, param):
        global selected_y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_y = y

    cv2.setMouseCallback(WIN, mouse_callback)

    #Color palette and UI constants
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
Function: Creating interactive lake reflection effect
- User clicks to set waterline
- Applies perspective warp, ripple distortion, blur, and darkening to create a realistic reflection
- Blends the reflection with the original image and saves the result

Parameters:
- perspective_shrink: Hhow much the reflection narrows towards the bottom to enhance perspective
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

    #Applied vertical compression to enhance perspective effect
    compressed_h = max(1, int(src_h * vertical_compression))
    reflected = cv2.resize(reflected, (w, compressed_h), interpolation=cv2.INTER_LINEAR)

    #Applied perspective warp to create a more realistic reflection
    inset = int(w * perspective_shrink / 2)
    src = np.float32([[0, 0], [w-1, 0], [0, compressed_h-1], [w-1, compressed_h-1]])
    dst = np.float32([[0, 0], [w-1, 0], [inset, compressed_h-1], [w-1-inset, compressed_h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    reflected = cv2.warpPerspective(reflected, M, (w, compressed_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)

    #Vectorized ripple distortion to simulate water surface
    ys = np.arange(compressed_h, dtype=np.float32)
    local_amp = wave_amp * (0.25 + 0.75 * (ys / max(compressed_h - 1, 1)))
    shifts    = (local_amp * np.sin(2 * np.pi * wave_freq * ys)).reshape(-1, 1)

    xs = np.tile(np.arange(w, dtype=np.float32), (compressed_h, 1))
    map_x = np.clip(xs + shifts, 0, w - 1)
    map_y = np.tile(ys.reshape(-1, 1), (1, w))

    reflected = cv2.remap(reflected, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    #Applying blur to soften the reflection for realism
    if blur_size % 2 == 0:
        blur_size += 1
    reflected = cv2.GaussianBlur(reflected, (blur_size, blur_size), 0)

    #Darkening + vertical fade for depth effect
    reflected = np.clip(reflected.astype(np.float32) * darken, 0, 255)
    fade = np.linspace(1.0, fade_min, compressed_h).reshape(compressed_h, 1, 1)
    reflected = np.clip(reflected * fade, 0, 255).astype(np.uint8)

    #Compile original and water-reflected parts as one
    final_h = reflection_start_y + compressed_h
    output  = np.zeros((final_h, w, 3), dtype=np.uint8)
    output[:reflection_start_y] = top_part
    output[reflection_start_y:reflection_start_y + compressed_h] = reflected

    #Waterline blending to hide seam between original and reflection
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

    #Subtle horizontal line at waterline for added definition
    cv2.line(output, (0, reflection_start_y), (w, reflection_start_y), (35, 35, 35), 1)

    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved → {output_path}")

"""
MAIN:
- Directory setup for input images and output results
- Pass parameters for water reflection effect
"""
if __name__ == "__main__":
    #Directory setup for input images and output results
    IMAGES_DIR  = Path("images")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    #Support any image formatting to allow flexibility for user input
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    #Image selection via user input
    image_input = input("Enter the name of your image (without extension): ").strip()

    #Check for valid input
    #Default to ferrarri.jpg if no input was provided
    if image_input == "":
        image_path = IMAGES_DIR / "ferrari.jpg"
    else:
        #Image searching with supported format
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

    #Output filename with timestep to avoid overwrite
    stem = image_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{stem}_reflection_{timestamp}.jpg"

    #Pass parameters to create interactive lake reflection
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