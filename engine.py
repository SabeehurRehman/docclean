"""Document cleanup engine: find the page, crop away the background,
straighten it, and make the text sharp and readable."""
from __future__ import annotations

import os
import sys
import threading

import cv2
import numpy as np

MODES = ["Enhanced color", "Grayscale", "Black & white", "Original (crop only)"]


# ---------------------------------------------------------------- I/O
def load_image(path: str) -> np.ndarray:
    """Read an image (works with non-ASCII Windows paths). Returns BGR."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to Pillow for formats OpenCV can't read (HEIC plugins etc.)
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        img = _apply_exif_rotation(path, img)
    return img


def _apply_exif_rotation(path: str, img: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
        orient = Image.open(path).getexif().get(0x0112, 1)
    except Exception:
        return img
    if orient == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orient == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orient == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def save_image(path: str, img: np.ndarray) -> None:
    ext = "." + path.rsplit(".", 1)[-1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)


def save_pdf(path: str, images: list[np.ndarray], dpi: int = 200) -> None:
    from PIL import Image
    pages = []
    for im in images:
        if im.ndim == 2:
            pages.append(Image.fromarray(im).convert("L"))
        else:
            pages.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=dpi)


# ---------------------------------------------------------------- detection
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def full_frame(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)


def _quad_from_mask(mask: np.ndarray, min_area: float):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
        # Not a clean 4-gon (e.g. a rounded or partly covered corner): use
        # the tightest rotated rectangle around it.
        return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return None


def _edge_maps(bgr: np.ndarray):
    """Per-channel Lab gradients (colour edges count, not just brightness)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1:] *= 2.0  # chroma channels have a smaller range than L
    gx = np.stack([cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)])
    gy = np.stack([cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)])
    return gx, gy


def _color_edges(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    return cv2.Canny(L, 20, 60) | cv2.Canny(A, 8, 24) | cv2.Canny(B, 8, 24)


def _quad_support(quad: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                  thr: float = 50.0, n: int = 48) -> tuple[float, float]:
    """How well the 4 sides of `quad` sit on real (colour) edges.
    Returns (mean side support, weakest side support), each 0..1."""
    h, w = gx.shape[1:]
    sides = []
    offs = np.arange(-3, 4, dtype=np.float32)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.hypot(*d))
        if length < 5:
            return 0.0, 0.0
        u = d / length
        nrm = np.array([-u[1], u[0]], np.float32)
        t = np.linspace(0.08, 0.92, n, dtype=np.float32)
        pts = a + t[:, None] * d
        samp = pts[:, None, :] + offs[None, :, None] * nrm
        xs = np.clip(np.round(samp[..., 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(samp[..., 1]).astype(int), 0, h - 1)
        inside = (samp[..., 0] >= 0) & (samp[..., 0] < w) & (samp[..., 1] >= 0) & (samp[..., 1] < h)
        across = np.sqrt(((gx[:, ys, xs] * nrm[0] + gy[:, ys, xs] * nrm[1]) ** 2).sum(axis=0)) * inside
        hit = across.max(axis=1) > thr
        out = ~inside.any(axis=1)
        sides.append((hit.sum() + 0.5 * out.sum()) / n)
    sides = np.array(sides)
    return float(sides.mean()), float(sides.min())


def _line_candidates(edges: np.ndarray, max_lines: int = 32):
    """Long straight lines (theta, rho, length) found in an edge map."""
    h, w = edges.shape
    md = min(h, w)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                           minLineLength=int(0.12 * md), maxLineGap=int(0.02 * md) + 3)
    if segs is None:
        return []
    lines = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(np.float32):
        L = float(np.hypot(x2 - x1, y2 - y1))
        theta = np.arctan2(y2 - y1, x2 - x1) % np.pi
        nx, ny = -np.sin(theta), np.cos(theta)
        rho = x1 * nx + y1 * ny
        lines.append([theta, rho, L])
    lines.sort(key=lambda l: -l[2])
    merged = []
    for th, rho, L in lines:
        for m in merged:
            dth = abs(th - m[0])
            if dth > np.pi / 2:  # wrap-around: same line, flipped normal
                dth = np.pi - dth
                rho_cmp = -rho
            else:
                rho_cmp = rho
            if dth < np.radians(3) and abs(rho_cmp - m[1]) < 0.015 * md + 4:
                m[2] += L
                break
        else:
            merged.append([th, rho, L])
    merged.sort(key=lambda l: -l[2])
    # Keep the strongest lines of each orientation separately, otherwise
    # long text lines (one direction) crowd out the page's other two edges.
    horiz = [l for l in merged if min(l[0], np.pi - l[0]) < np.pi / 4]
    vert = [l for l in merged if min(l[0], np.pi - l[0]) >= np.pi / 4]
    return horiz[:max_lines // 2] + vert[:max_lines // 2]


def _intersect(l1, l2):
    t1, r1, _ = l1
    t2, r2, _ = l2
    a = np.array([[-np.sin(t1), np.cos(t1)], [-np.sin(t2), np.cos(t2)]])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, np.array([r1, r2]))


def _quads_from_lines(lines, w, h, min_area):
    def angdiff(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    md = min(w, h)
    # Near-parallel line pairs = candidate opposite sides of the page.
    pairs = [(lines[i], lines[j]) for i in range(len(lines)) for j in range(i + 1, len(lines))
             if angdiff(lines[i][0], lines[j][0]) < np.radians(20)]
    quads = []
    margin = 0.05 * max(w, h)
    for a1, a2 in pairs:
        for b1, b2 in pairs:
            if angdiff(a1[0], b1[0]) < np.radians(55):
                continue
            pts = [_intersect(a1, b1), _intersect(a1, b2), _intersect(a2, b2), _intersect(a2, b1)]
            if any(p is None for p in pts):
                continue
            q = np.array(pts, np.float32)
            if (q[:, 0] < -margin).any() or (q[:, 0] > w + margin).any() or \
               (q[:, 1] < -margin).any() or (q[:, 1] > h + margin).any():
                continue
            q = order_corners(q)
            if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
                continue
            area = cv2.contourArea(q)
            if area < min_area:
                continue
            sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
            if min(sides) < 0.12 * md:
                continue
            quads.append(q)
    return quads


def _corner_angles_ok(q: np.ndarray, tol_deg: float = 30) -> bool:
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        if abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90) > tol_deg:
            return False
    return True


def _grabcut_quad(small: np.ndarray, min_area: float):
    """Separate the centre object from its surroundings by colour."""
    h, w = small.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[int(.3 * h):int(.7 * h), int(.3 * w):int(.7 * w)] = cv2.GC_PR_FGD
    b = max(3, int(.03 * min(h, w)))
    mask[:b] = mask[-b:] = cv2.GC_BGD
    mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    try:
        cv2.grabCut(small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return _quad_from_mask(fg, min_area)


def _detect_classic(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (corners in full-res coordinates, found?).

    The photo is first flattened with mean-shift filtering, which wipes out
    fabric/wood texture and text but keeps the document's outline. Candidate
    outlines then come from contours, long straight lines and colour
    segmentation; each is scored by how well all four sides sit on real edges."""
    h, w = img.shape[:2]
    scale = min(1.0, 560.0 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    sh, sw = small.shape[:2]
    min_area = 0.08 * sh * sw

    flat = cv2.pyrMeanShiftFiltering(small, 12, 40)
    fgray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    gx, gy = _edge_maps(flat)

    candidates = []
    # 1) Contours of the flattened image's edges
    for lo, hi in ((20, 60), (40, 120)):
        edges = cv2.Canny(fgray, lo, hi)
        q = _quad_from_mask(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2), min_area)
        if q is not None:
            candidates.append(q)
    # 2) Brightness / low saturation (paper vs. desk)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    paperness = cv2.subtract(hsv[:, :, 2], hsv[:, :, 1])
    _, th = cv2.threshold(paperness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    q = _quad_from_mask(th, min_area)
    if q is not None:
        candidates.append(q)
    # 3) Colour segmentation of the centre object
    q = _grabcut_quad(flat, min_area)
    if q is not None:
        candidates.append(q)
    # 4) Quadrilaterals made of long straight lines
    candidates += _quads_from_lines(_line_candidates(_color_edges(flat)), sw, sh, min_area)

    best, best_score = None, 0.0
    for q in candidates:
        q = order_corners(q)
        area = cv2.contourArea(q) / (sh * sw)
        if area > 0.97 or not _corner_angles_ok(q):
            continue
        mean_s, min_s = _quad_support(q, gx, gy)
        if min_s < 0.3:
            continue
        # Mostly edge evidence; mild preference for bigger outlines so an
        # inner box on the document doesn't beat the document's own border.
        s = (0.6 * mean_s + 0.4 * min_s) * (0.7 + 0.3 * np.sqrt(area))
        if s > best_score:
            best, best_score = q, s

    if best is None or best_score < 0.4:
        return full_frame(img), False
    corners = order_corners(best / scale)
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners, True


# ---------------------------------------------------------------- AI detection
_net = None
_net_lock = threading.Lock()


def _model_path() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", "u2netp.onnx")


def _get_net():
    global _net
    if _net is None:
        path = _model_path()
        if not os.path.exists(path):
            return None
        _net = cv2.dnn.readNetFromONNX(path)
    return _net


def _object_mask(img: np.ndarray) -> np.ndarray | None:
    """U2-Net salient-object mask (0..1, 320x320) – finds 'the thing in the photo'."""
    with _net_lock:
        net = _get_net()
        if net is None:
            return None
        x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
        x /= max(float(x.max()), 1.0)
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        net.setInput(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        out = net.forward()
    out = np.squeeze(out[0] if isinstance(out, (tuple, list)) else out)
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def _fit_quad_to_contour(cnt: np.ndarray) -> np.ndarray | None:
    """4 corners for a (possibly round-cornered) outline: find the 4 sides,
    fit a straight line to each, and intersect neighbouring lines."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a.reshape(4, 2).astype(np.float32)
            break
    if approx is None:
        approx = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    approx = order_corners(approx)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = approx[i], approx[(i + 1) % 4]
        d = b - a
        L = float(np.hypot(*d))
        if L < 4:
            return approx
        u = d / L
        nrm = np.array([-u[1], u[0]])
        rel = pts - a
        along = rel @ u
        dist = np.abs(rel @ nrm)
        # points on the straight middle part of this side (skip rounded corners)
        sel = pts[(along > 0.15 * L) & (along < 0.85 * L) & (dist < 0.04 * L + 3)]
        if len(sel) < 5:
            return approx
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return approx
        t = np.linalg.solve(A, p2 - p1)
        corners.append(p1 + t[0] * d1)
    q = order_corners(np.array(corners, np.float32))
    # sanity: refined quad must stay close to the rough one
    if np.max(np.linalg.norm(q - approx, axis=1)) > 0.15 * np.sqrt(cv2.contourArea(approx)):
        return approx
    return q


def _detect_ai(img: np.ndarray) -> np.ndarray | None:
    prob = _object_mask(img)
    if prob is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, 1000.0 / max(h, w))
    sw, sh = int(round(w * scale)), int(round(h * scale))
    m = (cv2.resize(prob, (sw, sh), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8) * 255
    k = max(3, int(0.01 * max(sw, sh)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if not (0.05 * sw * sh < area < 0.97 * sw * sh):
        return None
    q = _fit_quad_to_contour(cnt)
    # The object must actually look like a flat page: the 4-sided outline
    # should cover it closely (rejects e.g. a hand or a crumpled shape).
    poly = np.zeros_like(m)
    cv2.fillConvexPoly(poly, q.astype(np.int32), 255)
    blob = np.zeros_like(m)
    cv2.drawContours(blob, [cnt], -1, 255, -1)
    inter = np.count_nonzero(poly & blob)
    union = np.count_nonzero(poly | blob)
    if union == 0 or inter / union < 0.85 or not _corner_angles_ok(q, 35):
        return None
    q = q / scale
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q


def detect_document(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (corners in full-res coordinates, found?). Uses the AI model
    when available and confident, otherwise classic edge detection."""
    try:
        q = _detect_ai(img)
    except cv2.error:
        q = None
    if q is not None:
        return q, True
    return _detect_classic(img)


# ---------------------------------------------------------------- warp
def warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = order_corners(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 10), max(height, 10)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- enhance
def _upscale_if_small(img: np.ndarray, target: int = 1800) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) < target:
        f = target / max(h, w)
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return img


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels that are plain paper (bright, not ink / photos / logos)."""
    _, m = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.erode(m, np.ones((5, 5), np.uint8))


def _flatten_background(channel: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting/shadows so the paper becomes even white.

    Lighting is estimated only from plain-paper pixels and smoothly filled in
    elsewhere, so photos, logos and large dark areas keep their tones."""
    h, w = channel.shape[:2]
    f = min(1.0, 400.0 / max(h, w))
    small = cv2.resize(channel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(paper, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
    sigma = max(small.shape) / 12.0
    num = cv2.GaussianBlur(small * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    fallback = float(np.percentile(small, 90))
    bg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    out = channel.astype(np.float32) / np.maximum(bg, 1.0) * 245.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 1.6)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _levels(img: np.ndarray, black: int = 30, white: int = 235) -> np.ndarray:
    lut = np.clip((np.arange(256) - black) * 255.0 / (white - black), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def enhance(img: np.ndarray, mode: str, sharpness: float = 1.0) -> np.ndarray:
    """sharpness: 0 (none) .. 2 (strong)."""
    if mode == MODES[3]:
        return img
    img = _upscale_if_small(img)
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21) if max(img.shape[:2]) < 2600 else img

    if mode == MODES[0]:  # Enhanced color
        paper = _paper_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        chans = [_flatten_background(c, paper) for c in cv2.split(img)]
        out = cv2.merge(chans)
        out = _levels(out, 20, 240)
        # Restore a bit of colour saturation lost in flattening.
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return _sharpen(out, 0.8 * sharpness)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = _flatten_background(gray, _paper_mask(gray))

    if mode == MODES[1]:  # Grayscale
        out = _levels(flat, 40, 235)
        return _sharpen(out, 0.9 * sharpness)

    # Black & white
    flat = _sharpen(flat, 0.6 * sharpness)
    block = max(15, (min(flat.shape) // 40) | 1)
    bw = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, 12)
    # Remove salt-and-pepper specks.
    bw = cv2.medianBlur(bw, 3)
    return bw


# ---------------------------------------------------------------- orientation
def _text_line_lengths(gray: np.ndarray) -> tuple[float, float]:
    """Total length of horizontal vs vertical text-line-like bars."""
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    k = max(9, int(0.03 * max(gray.shape)))
    res = []
    for horiz in (True, False):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if horiz else (1, k))
        sm = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ker)
        _, _, st, _ = cv2.connectedComponentsWithStats(sm)
        L = st[1:, 2] if horiz else st[1:, 3]
        T = st[1:, 3] if horiz else st[1:, 2]
        ok = (L > 4 * T) & (L > 0.08 * max(gray.shape))
        res.append(float((L * ok).sum()) + 1.0)
    return res[0], res[1]


def _profile_contrast(gray: np.ndarray) -> tuple[float, float]:
    """How strongly ink separates into rows vs. into columns."""
    ink = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15) > 0)
    m = int(0.06 * min(gray.shape))
    ink = ink[m:-m, m:-m].astype(np.float32) if m > 0 else ink.astype(np.float32)
    out = []
    for ax in (1, 0):
        p = cv2.GaussianBlur(ink.mean(axis=ax).reshape(-1, 1), (1, 5), 0).ravel()
        out.append(float(np.abs(np.diff(p)).sum() / (p.mean() + 1e-6) / len(p)) + 1e-3)
    return out[0], out[1]


def auto_rotation(page: np.ndarray) -> int:
    """Suggest a rotation (0 or 270 degrees) so text reads horizontally.

    Phone photos of landscape cards/pages often come out sideways. Two
    independent cues vote: long text-line shapes and row/column ink profiles."""
    gray = page if page.ndim == 2 else cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    f = 800.0 / max(gray.shape)
    gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h_len, v_len = _text_line_lengths(gray)
    rows, cols = _profile_contrast(gray)
    score = float(np.clip(np.log(v_len / h_len), -2, 2)) + 3.0 * float(np.log(cols / rows))
    hh, ww = gray.shape
    if hh > ww and 1.45 < hh / ww < 1.75:  # ID-card shaped and standing upright
        score += 0.5
    # Sideways text in these photos reads bottom-to-top, so turn it left.
    return 270 if score > 0.4 else 0


def suggest_rotation(img: np.ndarray, corners: np.ndarray) -> int:
    h, w = img.shape[:2]
    f = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    return auto_rotation(warp(small, corners * f))


def process(img: np.ndarray, corners: np.ndarray | None, mode: str,
            sharpness: float = 1.0, rotation: int = 0) -> np.ndarray:
    if corners is None:
        corners, _ = detect_document(img)
    flat = warp(img, corners)
    # Trim a hair off each edge so no sliver of desk/background survives.
    h, w = flat.shape[:2]
    m = max(2, int(0.006 * min(h, w)))
    if h > 4 * m and w > 4 * m:
        flat = flat[m:h - m, m:w - m]
    out = enhance(flat, mode, sharpness)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotation % 360 in rot:
        out = cv2.rotate(out, rot[rotation % 360])
    return out
