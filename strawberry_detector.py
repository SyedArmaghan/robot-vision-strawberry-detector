import os
import glob
import cv2
import numpy as np

def detect_and_classify_strawberries(img):
    h, w = img.shape[:2]
    # Standardize image width for consistent morphology
    target_w = 900
    scale = target_w / float(w)
    img_work = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(img_work.astype(np.int16))

    # =========================================================================
    # 1. VEGETATION-SUPPRESSION INDEX (R - G Difference)
    # Leaves: R - G < 0 -> clamped to 0 (pure black)
    # White Netting/PVC: R - G ~ 0 -> clamped to 0
    # Ripe Strawberries: R - G is strongly positive (40 to 180+)
    # =========================================================================
    rg_diff = r - g
    red_signal = np.clip(rg_diff, 0, 255).astype(np.uint8)

    # Threshold red signal to isolate ripe fruit
    _, ripe_mask = cv2.threshold(red_signal, 35, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ripe_mask = cv2.morphologyEx(ripe_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    ripe_mask = cv2.morphologyEx(ripe_mask, cv2.MORPH_OPEN, k_close, iterations=1)

    # =========================================================================
    # 2. RAW / UNRIPE DETECTION (Pale Yellowish-White Fruit in LAB)
    # Rejects dark green leaves and flat white plastic netting
    # =========================================================================
    lab = cv2.cvtColor(img_work, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    raw_mask = (L > 115) & (A >= 115) & (A <= 138) & (B > 132) & (red_signal < 30)
    raw_mask = np.uint8(raw_mask * 255)

    # Fruit has seed bumps/texture; smooth white plastic netting does not
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    _, texture_mask = cv2.threshold(sobel, 20, 255, cv2.THRESH_BINARY)
    raw_mask = cv2.bitwise_and(raw_mask, cv2.dilate(texture_mask, k_close))

    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_close, iterations=1)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    detections = []
    output_img = img_work.copy()

    # -------------------------------------------------------------------------
    # PROCESS RIPE / SEMI-RIPE STRAWBERRIES
    # -------------------------------------------------------------------------
    contours_ripe, _ = cv2.findContours(ripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_ripe:
        area = cv2.contourArea(cnt)
        if area < 350:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = float(bw) / bh
        if aspect < 0.35 or aspect > 2.6:
            continue

        # Measure average red intensity inside the contour
        c_mask = np.zeros(ripe_mask.shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(red_signal, mask=c_mask)[0]

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + bh // 2

        if mean_val >= 60:
            status = "Ripe"
            color = (0, 0, 255)  # Red bounding box
        else:
            status = "Semi-Ripe"
            color = (0, 255, 255)  # Yellow bounding box

        detections.append({
            "status": status,
            "score": round(mean_val, 1),
            "centroid": (int(cx / scale), int(cy / scale)),
            "bbox": (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
        })

        cv2.rectangle(output_img, (x, y), (x + bw, y + bh), color, 2)
        cv2.circle(output_img, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(output_img, f"{status}", (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

    # -------------------------------------------------------------------------
    # PROCESS RAW / UNRIPE STRAWBERRIES
    # -------------------------------------------------------------------------
    contours_raw, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 6000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = float(bw) / bh
        if aspect < 0.5 or aspect > 1.8:
            continue

        # Reject irregular leaf shapes with solidity check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0 or (float(area) / hull_area) < 0.70:
            continue

        # Prevent overlapping with already detected ripe fruit
        c_mask = np.zeros(ripe_mask.shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(ripe_mask, c_mask)) > 0.25 * area:
            continue

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + bh // 2

        detections.append({
            "status": "Raw",
            "score": 0.0,
            "centroid": (int(cx / scale), int(cy / scale)),
            "bbox": (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
        })

        cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(output_img, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(output_img, "Raw", (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

    return output_img, detections

def process_folder(input_folder, output_folder="output_results"):
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print(f"No images found in folder: '{input_folder}'")
        return

    print(f"Processing {len(image_paths)} images using vegetation-index suppression...\n")

    for idx, path in enumerate(image_paths, start=1):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        annotated_img, detections = detect_and_classify_strawberries(img)
        counts = {"Ripe": 0, "Semi-Ripe": 0, "Raw": 0}
        for d in detections:
            counts[d["status"]] += 1

        out_path = os.path.join(output_folder, f"result_{filename}")
        cv2.imwrite(out_path, annotated_img)
        print(f"[{idx}/{len(image_paths)}] {filename} -> Ripe: {counts['Ripe']} | Semi-Ripe: {counts['Semi-Ripe']} | Raw: {counts['Raw']}")

    print(f"\nAll images processed! Output saved in: '{os.path.abspath(output_folder)}'")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    process_folder(folder)