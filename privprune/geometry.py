"""Image <-> visual-token geometry for LLaVA-1.5.

LLaVA-1.5 pads the image to a square (filled with the CLIP mean colour), then resizes it to
336x336. CLIP ViT-L/14 cuts that into a 24x24 grid of 14x14 patches, flattened row-major, so
token i covers rows [14*(i//24), 14*(i//24)+14) and cols [14*(i%24), 14*(i%24)+14).

We apply the *same* transform to the privacy polygons, so the token-level overlap
o_{i,c} = |A_i ∩ M_c| / |A_i| is exact up to rasterisation.
"""
import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 336
PATCH = 14
GRID = IMAGE_SIZE // PATCH  # 24
NUM_TOKENS = GRID * GRID  # 576
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)


def pad_to_square(img: Image.Image, fill=CLIP_MEAN):
    """Pad like LLaVA-1.5 (`expand2square`). Returns padded image and the (dx, dy) offset."""
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    dx, dy = (s - w) // 2, (s - h) // 2
    canvas = Image.new("RGB", (s, s), tuple(int(255 * c) for c in fill))
    canvas.paste(img, (dx, dy))
    return canvas, (dx, dy)


def llava_square_image(img: Image.Image):
    """Padded + resized 336x336 image, plus the transform needed to map polygons into it."""
    w, h = img.size
    padded, (dx, dy) = pad_to_square(img)
    scale = IMAGE_SIZE / padded.size[0]
    return padded.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), (dx, dy, scale)


def transform_polygon(flat_xy, dx, dy, scale):
    """[x1, y1, x2, y2, ...] in original pixels -> [(x, y), ...] in 336x336 pixels."""
    xs = (np.asarray(flat_xy[0::2], dtype=np.float64) + dx) * scale
    ys = (np.asarray(flat_xy[1::2], dtype=np.float64) + dy) * scale
    return list(zip(xs.tolist(), ys.tolist()))


def rasterize(polygons, supersample=4):
    """Union of polygons (in 336-space) -> boolean mask at (336*ss, 336*ss)."""
    size = IMAGE_SIZE * supersample
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([(x * supersample, y * supersample) for x, y in poly], fill=1)
    return np.asarray(canvas, dtype=bool)


def mask_to_token_overlap(mask):
    """High-res boolean mask -> (576,) fraction of each patch covered by the mask."""
    s = mask.shape[0] // GRID
    return mask.reshape(GRID, s, GRID, s).mean(axis=(1, 3)).reshape(-1).astype(np.float32)


def token_to_pixel_box(i):
    r, c = divmod(int(i), GRID)
    return c * PATCH, r * PATCH, (c + 1) * PATCH, (r + 1) * PATCH


def render_kept_tokens(img336: Image.Image, keep_idx, dim=0.15):
    """Visualise a token selection: dropped patches are darkened."""
    arr = np.asarray(img336, dtype=np.float32)
    keep = np.zeros(NUM_TOKENS, dtype=bool)
    keep[np.asarray(keep_idx, dtype=int)] = True
    m = np.repeat(np.repeat(keep.reshape(GRID, GRID), PATCH, 0), PATCH, 1)
    arr = np.where(m[..., None], arr, arr * dim)
    return Image.fromarray(arr.astype(np.uint8))


def dilate_tokens(p, r, grid=24):
    """Grey-scale dilation of a per-token score on the grid: each token takes the max over its (2r+1)^2 neighbourhood."""
    m = np.asarray(p, dtype=np.float32).reshape(grid, grid)
    pad = np.pad(m, r, mode="constant")
    out = np.zeros_like(m)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out = np.maximum(out, pad[dy:dy + grid, dx:dx + grid])
    return out.reshape(-1)
