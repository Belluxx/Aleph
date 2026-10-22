"""Texture makers for library materials that need more than a downloaded set: paving and roof layouts built on
ambientCG maps, decals, sign and poster atlases, vehicle and vegetation textures.

Each maker takes (spec, n, maps, rng): the MATERIALS recipe, the texture size, ambientCG maps (Pillow images by map
name, or None offline) and a seeded numpy generator. It returns (colour, normal, orm) or (colour, normal, orm,
height) arrays: sRGB colour (RGBA when the material has alpha), an OpenGL normal map, glTF's packed occlusion/
roughness/metallic map, and an optional height field (any range, white = high once stored). Everything works
without maps, so offline builds still get every material.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .library import _blur, _fbm, _noise, _normal, _procedural, _tint, pack_orm

FLAT = (128, 128, 255)


# Shared helpers


def _resize(image, n, mode=None):
    image = image.convert(mode) if mode else image
    return image if image.size == (n, n) else image.resize((n, n), Image.LANCZOS)


def _surface(spec, n, maps, rng):
    """A material's base maps at n × n as floats: colour (sRGB 0–255), normal (−1–1), roughness, occlusion,
    metalness, and height, from ambientCG maps (tinted) or the procedural fallback recipe."""
    tint = spec.get("tint") or spec["fallback"]["color"]
    if maps:
        color = np.asarray(_resize(_tint(maps["Color"].convert("RGB"), tint), n), float)
        normal = np.asarray(_resize(maps["NormalGL"], n, "RGB"), float) / 127.5 - 1
        gray = lambda key, fill: (np.asarray(_resize(maps[key], n, "L"), float) / 255 if key in maps
                                  else np.full((n, n), fill))
        rough = np.clip(gray("Roughness", 0.8) * spec.get("rough", 1.0), 0, 1)
        metal = gray("Metalness", 0.0) if spec.get("metal") is None else np.full((n, n), float(spec["metal"]))
        height = gray("Displacement", 0.5)
        return dict(color=color, normal=normal, rough=rough, ao=gray("AmbientOcclusion", 1.0), metal=metal, height=height)
    color, normal, orm, height = _procedural(spec["fallback"], n, rng)
    orm = orm.astype(float) / 255
    return dict(color=color.astype(float), normal=normal.astype(float) / 127.5 - 1, rough=orm[..., 1], ao=orm[..., 0],
                metal=orm[..., 2], height=height / max(np.ptp(height), 1e-9))


def _detail(normal, height, strength):
    """Add a height field's bumps to a −1–1 normal map (whiteout blend); returns uint8."""
    extra = _normal(height, strength).astype(float) / 127.5 - 1
    n = np.stack((normal[..., 0] + extra[..., 0], normal[..., 1] + extra[..., 1], normal[..., 2] * extra[..., 2]), -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n + 1) * 127.5).clip(0, 255).astype(np.uint8)


def _u8(color):
    return np.clip(color, 0, 255).astype(np.uint8)


def _rgba(color, alpha):
    """RGBA uint8 from sRGB colour and 0–1 alpha, bleeding colour into transparent texels so mipmaps don't fringe."""
    color = np.asarray(color, float)
    alpha = np.clip(np.asarray(alpha, float), 0, 1)
    weight = _blur(alpha, 4) + 1e-6
    bled = np.stack([_blur(color[..., k] * alpha, 4) for k in range(3)], -1) / weight[..., None]
    color = color * alpha[..., None] + bled * (1 - alpha[..., None])
    return np.dstack((_u8(color), _u8(alpha * 255)))


def _font(size, serif=False, bold=True):
    names = (["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "FreeSerifBold.otf", "georgiab.ttf"] if serif else
             ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.otf", "arialbd.ttf"])
    if not bold:
        names = [name.replace("-Bold", "").replace("Bold", "").replace("bd", "") for name in names]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _stretched(rng, shape, cells=(4, 32), octaves=4):
    """Fractal noise of shape (rows, cols), stretched along rows (streaks), tileable along columns."""
    rows, cols = shape
    n = max(rows, cols)
    noise = _fbm(rng, n, cells[1], octaves)
    return np.asarray(Image.fromarray(noise[: max(n // cells[1] * cells[0], 2)].astype(np.float32), "F")
                      .resize((cols, rows), Image.BICUBIC), float)


def _radial(n, cx=0.5, cy=0.5):
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    return np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)


# Paving and roofs


def _make_slabs(spec, n, maps, rng):
    """Large granite slabs in 0.75 m courses of mixed lengths (running bond), each with its own tint, texture offset,
    slight tilt, and a few dark gum spots; thin sanded joints. Repeats every spec size metres."""
    size = spec["size"]
    px = n / size  # Pixels per metre.
    granite = _surface(spec, n // 4, maps, rng)  # One granite repeat per quarter of the tile.
    course = 0.75
    rows = int(round(size / course))
    y, x = np.mgrid[0:n, 0:n]
    row = (y / px // course).astype(int) % rows
    slab_id = np.zeros((n, n), int)
    u0 = np.zeros((n, n))
    next_id = 0
    for r in range(rows):
        lengths, total = [], 0.0
        while total < size - 1e-6:
            length = rng.choice([0.75, 1.0, 1.25, 1.5])
            length = min(length, size - total) if size - total - length < 0.5 else length
            lengths.append(length)
            total += length
        start = rng.uniform(0, size)
        inside = row == r
        xm = x[inside] / px
        rel = (xm - start) % size
        k = np.searchsorted(np.cumsum(lengths), rel, side="right")
        slab_id[inside] = next_id + k
        u0[inside] = rel - np.concatenate(([0], np.cumsum(lengths)))[k]
        next_id += len(lengths)
    v0 = (y / px) % course
    # Distance to the nearest joint (in metres) from the slab's own coordinates.
    length = np.zeros((n, n))
    for r in range(rows):
        inside = row == r
        rel_ids = slab_id[inside]
        # Slab length from the min/max u of each id.
        ids, inverse = np.unique(rel_ids, return_inverse=True)
        maxu = np.zeros(len(ids))
        np.maximum.at(maxu, inverse, u0[inside])
        length[inside] = maxu[inverse] + 1 / px
    edge = np.minimum(np.minimum(u0, length - u0), np.minimum(v0, course - v0))
    joint = 0.004
    face = np.clip((edge - joint) / 0.006, 0, 1)  # Small arris (bevel) at slab edges.
    count = next_id
    tint = 1 + rng.normal(0, 0.045, count)
    warm = rng.normal(0, 0.02, count)
    offset = rng.integers(0, len(granite["color"]), (count, 2))
    gy = (y + offset[slab_id, 0]) % len(granite["color"])
    gx = (x + offset[slab_id, 1]) % len(granite["color"])
    color = granite["color"][gy, gx] * tint[slab_id][..., None]
    color[..., 0] *= 1 + warm[slab_id]
    color[..., 2] *= 1 - warm[slab_id]
    # Wear: a broad, faint darkening along joints and in large blotches (water, dirt).
    blotch = _fbm(rng, n, 4, 5)
    color *= (0.94 + 0.08 * blotch)[..., None]
    sand = np.array([118, 112, 102.0]) * (0.85 + 0.3 * _fbm(rng, n, 64, 2))[..., None]
    dirt = np.clip(1 - edge / 0.03, 0, 1) * 0.12
    color = color * (1 - dirt[..., None]) * face[..., None] + sand * (1 - face[..., None])
    # Gum spots: small dark discs, a few per slab.
    spots = np.zeros((n, n))
    for _ in range(int(size * size * 3)):
        cx, cy, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0.006, 0.012) * px
        d = np.hypot((x - cx + n / 2) % n - n / 2, (y - cy + n / 2) % n - n / 2)
        spots = np.maximum(spots, np.clip(1 - d / r, 0, 1) ** 0.5 * rng.uniform(0.3, 0.7))
    color = color * (1 - spots[..., None]) + np.array([62, 60, 58.0]) * spots[..., None]
    tilt = rng.normal(0, 1, (count, 2))
    height = face * (1 + (tilt[slab_id, 0] * (u0 - length / 2) + tilt[slab_id, 1] * (v0 - course / 2)) * 0.01) \
        + granite["height"][gy, gx] * 0.02
    normal = np.asarray(granite["normal"])[gy, gx]
    normal = _detail(normal, height * px * 0.004, 1.0)
    rough = np.where(face > 0.5, granite["rough"][gy, gx], 0.95) * (1 - 0.25 * spots)
    occlusion = 0.75 + 0.25 * face
    return _u8(color), normal, pack_orm(occlusion, rough, 0.0), height


def _make_seams(spec, n, maps, rng):
    """Standing-seam or lapped sheet roofing: raised ribs every 1/seams of the tile, slightly different pans,
    oil-canning waves and run-off streaks; on a downloaded metal (or patina) when available."""
    recipe = spec["fallback"]
    seams = recipe["seams"]
    base = _surface(dict(spec, tint=spec.get("tint") or recipe["color"]), n, maps, rng) if maps else None
    x = (np.arange(n) / n * seams) % 1
    pan = (np.arange(n) / n * seams).astype(int)
    width = 0.012 * seams  # Rib about 3–4 cm wide at 3 m per tile.
    ridge = np.clip(1 - np.abs(x - 0.5) / width, 0, 1) ** 0.6
    shoulder = np.exp(-((np.abs(x - 0.5) - width) / (width * 0.6)) ** 2) * 0.15
    streaks = _stretched(rng, (n, n), (2, 48)) * 0.6 + _fbm(rng, n, 4) * 0.4
    shade = 1 + rng.normal(0, 0.035, seams)[pan][None, :]
    canning = np.sin((np.arange(n)[:, None] / n * 7 + rng.random(seams)[pan][None, :] * 6) * np.pi) * 0.5 + 0.5
    if base is not None:
        color = base["color"]
        rough = base["rough"]
        metal = base["metal"] if spec.get("metal") is None else np.full((n, n), float(spec["metal"]))
        normal = base["normal"]
    else:
        color = np.broadcast_to(np.asarray(recipe["color"], float), (n, n, 3)).copy()
        rough = np.full((n, n), recipe["rough"])
        metal = np.full((n, n), recipe["metal"])
        normal = np.zeros((n, n, 3))
        normal[..., 2] = 1
    color = color * shade[..., None] * (0.9 + 0.2 * streaks[..., None]) * (1 + 0.08 * ridge[None, :, None])
    height = np.broadcast_to(ridge * 1.0 + shoulder, (n, n)) + canning * 0.03 * (1 - ridge[None, :])
    rough = np.clip(rough * (0.85 + 0.3 * streaks), 0, 1)
    if maps is None:
        rough = np.clip(recipe["rough"] * (0.85 + 0.3 * streaks), 0, 1)
    orm = pack_orm(1 - 0.2 * shoulder[None, :] * 4, rough, metal)
    return _u8(color), _detail(normal, height * 6, 1.0), orm, height


def _make_treegrate(spec, n, maps, rng):
    """A 1.4 m square cast-iron tree grate (unit UVs): a frame and rings of radial slots around a round opening with
    soil and mulch; the soil shows dark through the slots."""
    r, angle = _radial(n)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    soil = _surface(dict(spec, tint=(70, 56, 42)), n, maps, rng) if maps else None
    mulch = _fbm(rng, n, 48, 3)
    soil_color = soil["color"] * 0.8 if soil is not None else np.array([70, 56, 42.0]) * (0.7 + 0.5 * mulch)[..., None]
    border = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    frame = border < 0.035
    opening = r < 0.2
    rings = np.floor((r - 0.2) / 0.055).astype(int)
    ring_pos = ((r - 0.2) / 0.055) % 1
    slots_per_ring = 24 + rings * 8
    slot_pos = (angle / (2 * np.pi) * slots_per_ring) % 1
    slot = (ring_pos > 0.22) & (ring_pos < 0.78) & (slot_pos > 0.18) & (slot_pos < 0.82) & (r < 0.47) & ~opening
    iron = ~(opening | slot)
    worn = _fbm(rng, n, 16, 4)
    rust = np.clip(_fbm(rng, n, 8, 4) * 1.6 - 0.7, 0, 1)
    iron_color = np.array([54, 52, 50.0]) * (0.85 + 0.35 * worn[..., None])
    iron_color = iron_color * (1 - rust[..., None] * 0.6) + np.array([98, 64, 40.0]) * rust[..., None] * 0.6
    edge = _blur(iron.astype(float), 2)
    iron_color *= (0.8 + 0.25 * edge)[..., None]  # Raised edges catch light; recesses darken.
    color = np.where(iron[..., None], iron_color, soil_color * np.where(slot, 0.45, 1.0)[..., None])
    color = np.where(frame[..., None], iron_color * 1.05, color)
    height = np.where(iron, 1.0, np.where(opening, 0.35 + 0.15 * mulch, 0.0))
    rough = np.where(iron, 0.55 + 0.3 * rust, 0.95)
    metal = np.where(iron, 0.45 * (1 - rust), 0.0)
    occlusion = np.where(slot, 0.4, 0.75 + 0.25 * _blur(height, 3))
    return _u8(color), _normal(_blur(height, 1) * 10, 1.0), pack_orm(occlusion, rough, metal), height


def _make_dressed(spec, n, maps, rng):
    """Dressed (tooled) fine sandstone: the base set's grain flattened, gentle warm mottling, and faint parallel chisel
    marks; much smoother than a natural rock face."""
    base = _surface(spec, n, maps, rng)
    grain = _fbm(rng, n, 128, 2)
    mottle = _fbm(rng, n, 3, 4)
    y, x = np.mgrid[0:n, 0:n] / n
    tooling = np.sin((x + y * 0.35) * n / 5.0 * np.pi + _noise(rng, n, 16) * 3) * 0.5 + 0.5
    tooling *= 0.4 + 0.6 * _fbm(rng, n, 8, 3)
    gray = base["color"].mean(-1, keepdims=True)
    color = (gray / max(float(gray.mean()), 1) * 0.35 + 0.65) * np.asarray(spec["tint"], float)
    color = color * (0.95 + 0.07 * mottle[..., None] + 0.03 * grain[..., None] - 0.02 * tooling[..., None])
    flat = base["normal"].copy()
    flat[..., :2] *= 0.3
    height = grain * 0.4 + tooling * 0.5
    return _u8(color), _detail(flat, height, 0.8), pack_orm(0.95 + 0.05 * mottle, 0.72 + 0.1 * grain, 0.0), height


# Fabric, signs, and posters


def _make_awning(spec, n, maps, rng):
    """Near-white canvas for awnings (unit UVs): the weave tiled 4 × 4, faint panel seams, and a little dirt."""
    cloth = _surface(dict(spec, tint=spec["fallback"]["color"]), n // 4, maps, rng)
    tile = lambda a: np.tile(a, (4, 4) + (1,) * (a.ndim - 2))
    color, normal, rough = tile(cloth["color"]), tile(cloth["normal"]), tile(cloth["rough"])
    x = np.arange(n) / n
    seam = np.exp(-((((x * 4) % 1) - 0.5) * n / 4 / 2.5) ** 2)
    dirt = _fbm(rng, n, 4, 5)
    color = color * (1 - 0.06 * seam[None, :, None]) * (0.95 + 0.07 * dirt[..., None])
    normal = _detail(normal, np.broadcast_to(seam * 2, (n, n)), 1.0)
    return _u8(color), normal, pack_orm(1 - 0.15 * seam[None, :], np.clip(rough * 1.05, 0.6, 1), 0.0)


SHOP_SIGNS = [  # (word, background, letters, serif, metallic letters)
    ("CAFÉ", (32, 52, 42), (226, 204, 150), True, True), ("BÄCKEREI", (236, 228, 208), (110, 60, 30), True, False),
    ("BLUMEN", (40, 70, 52), (240, 238, 228), False, False), ("FRISEUR", (22, 22, 24), (232, 232, 232), False, False),
    ("SPÄTKAUF", (150, 28, 30), (250, 246, 236), False, False), ("BUCHHANDLUNG", (28, 40, 70), (226, 200, 130), True, True),
    ("OPTIK", (230, 230, 228), (30, 30, 34), False, False), ("IMBISS", (200, 140, 20), (40, 24, 12), False, False),
    ("WEINHANDLUNG", (70, 22, 34), (226, 200, 140), True, True), ("RESTAURANT", (24, 24, 26), (214, 180, 110), True, True),
    ("MODE", (240, 240, 240), (20, 20, 20), False, False), ("BIOMARKT", (96, 140, 60), (250, 250, 240), False, False),
    ("GALERIE", (250, 250, 248), (40, 40, 40), True, False), ("REISEBÜRO", (24, 90, 150), (250, 250, 250), False, False),
    ("ZEITUNGEN", (200, 196, 186), (30, 30, 30), True, False), ("SCHUHE", (60, 46, 36), (236, 220, 190), True, True),
]


def _make_shopsigns(spec, n, maps, rng):
    """2 × 8 atlas of 4:1 shop fascia boards with generic trade names (no brands): painted boards, raised letters
    (some brass), a border, and slight weathering."""
    cols, rows = spec["extras"]["atlas"]
    cw, ch = n // cols, n // rows
    color = Image.new("RGB", (n, n))
    relief = Image.new("L", (n, n), 0)
    metal = Image.new("L", (n, n), 0)
    draw, rdraw, mdraw = ImageDraw.Draw(color), ImageDraw.Draw(relief), ImageDraw.Draw(metal)
    for k, (word, back, ink, serif, brass) in enumerate(SHOP_SIGNS[: cols * rows]):
        x0, y0 = k % cols * cw, k // cols * ch
        draw.rectangle((x0, y0, x0 + cw, y0 + ch), fill=back)
        rdraw.rectangle((x0, y0, x0 + cw, y0 + ch), fill=60)
        inset = int(ch * 0.08)
        line = tuple(int(c * 0.7 + i * 0.3) for c, i in zip(back, ink))
        draw.rectangle((x0 + inset, y0 + inset, x0 + cw - inset, y0 + ch - inset), outline=line, width=max(2, ch // 40))
        rdraw.rectangle((x0, y0, x0 + cw, y0 + ch), outline=110, width=inset // 2)
        size = int(ch * 0.5)
        font = _font(size, serif)
        while font.getbbox(word)[2] > cw * 0.82 and size > 10:
            size -= 4
            font = _font(size, serif)
        centre = (x0 + cw / 2, y0 + ch / 2)
        draw.text((centre[0] + 2, centre[1] + 3), word, fill=tuple(int(c * 0.55) for c in back), font=font, anchor="mm")
        draw.text(centre, word, fill=ink, font=font, anchor="mm")
        rdraw.text(centre, word, fill=255, font=font, anchor="mm")
        if brass:
            mdraw.text(centre, word, fill=255, font=font, anchor="mm")
    color = np.asarray(color, float)
    relief = np.asarray(relief, float) / 255
    wear = _fbm(rng, n, 16, 4)
    color *= (0.9 + 0.12 * wear)[..., None]
    metallic = np.asarray(metal, float) / 255
    rough = np.where(metallic > 0.5, 0.35, 0.55 + 0.2 * wear)
    return _u8(color), _normal(_blur(relief, 1) * 6, 1.0), pack_orm(0.8 + 0.2 * relief, rough, metallic)


POSTER_WORDS = ["KONZERT", "AUSSTELLUNG", "THEATER", "FESTIVAL", "LESUNG", "OPER", "KINO", "TANZ", "JAZZ", "ZIRKUS"]


def _make_posters(spec, n, maps, rng):
    """Advertising-column band (unit UVs, u once around): two rows of five generic event posters with abstract
    graphics and nonsense type, pasted paper with seams, wrinkles, and torn corners. No brands. Drawn at the band's
    true aspect (a 1.3 m column is 4.08 m around, the band 2.4 m tall) and squeezed into the square texture."""
    w, h = n, int(n * 2.4 / 4.08)
    image = Image.new("RGB", (w, h), (226, 222, 210))
    relief = Image.new("L", (w, h), 128)
    draw, rdraw = ImageDraw.Draw(image), ImageDraw.Draw(relief)
    palette = [(214, 48, 49), (9, 132, 227), (253, 203, 110), (0, 184, 148), (45, 52, 54), (232, 67, 147),
               (108, 92, 231), (250, 177, 160), (255, 118, 117), (85, 239, 196), (240, 240, 232), (30, 30, 30)]
    pw, ph = w // 5, h // 2
    for k in range(10):
        x0, y0 = k % 5 * pw, k // 5 * ph
        pad = int(pw * 0.02)
        box = (x0 + pad, y0 + pad, x0 + pw - pad, y0 + ph - pad)
        bg, fg, accent = (palette[i] for i in rng.choice(len(palette), 3, replace=False))
        draw.rectangle(box, fill=bg)
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        style = k % 4
        if style == 0:  # Big circle and rings.
            cx, cy, r = bx0 + bw * rng.uniform(0.3, 0.7), by0 + bh * 0.4, bw * rng.uniform(0.25, 0.4)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
            draw.ellipse((cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5), fill=accent)
        elif style == 1:  # Diagonal bands.
            for j in range(5):
                o = bw * (j * 0.35 - 0.4)
                draw.polygon([(bx0 + o, by1), (bx0 + o + bw * 0.15, by1), (bx0 + o + bw * 0.9, by0), (bx0 + o + bw * 0.75, by0)],
                             fill=fg if j % 2 else accent)
            draw.rectangle(box, outline=bg, width=pad * 2)
        elif style == 2:  # Photo-like gradient block with a silhouette.
            grad = np.linspace(0, 1, int(bh * 0.6))[:, None, None]
            block = np.asarray(fg, float) * (1 - grad) + np.asarray(accent, float) * grad
            block = np.broadcast_to(block, (len(grad), bw, 3)).astype(np.uint8)
            image.paste(Image.fromarray(block), (bx0, by0))
            draw.ellipse((bx0 + bw * 0.35, by0 + bh * 0.15, bx0 + bw * 0.65, by0 + bh * 0.33), fill=bg)
            draw.rectangle((bx0 + bw * 0.3, by0 + bh * 0.33, bx0 + bw * 0.7, by0 + bh * 0.6), fill=bg)
        else:  # Grid of squares.
            for i in range(4):
                for j in range(5):
                    if rng.random() < 0.6:
                        s = bw / 4
                        draw.rectangle((bx0 + i * s + 3, by0 + j * s * 0.9 + 3, bx0 + (i + 1) * s - 3, by0 + (j + 1) * s * 0.9 - 3),
                                       fill=fg if rng.random() < 0.5 else accent)
        word = POSTER_WORDS[k]
        size = int(bw * 0.16)
        font = _font(size, serif=k % 3 == 0)
        while font.getbbox(word)[2] > bw * 0.88 and size > 8:
            size -= 2
            font = _font(size, serif=k % 3 == 0)
        ink = (20, 20, 20) if sum(bg) > 420 else (245, 245, 240)
        draw.text((bx0 + bw / 2, by0 + bh * 0.74), word, fill=ink, font=font, anchor="mm")
        small = _font(max(8, size // 3), bold=False)
        for j in range(3):
            draw.text((bx0 + bw / 2, by0 + bh * (0.84 + j * 0.045)), "lorem ipsum dolor sit" if j % 2 else "amet 19:30",
                      fill=ink, font=small, anchor="mm")
        rdraw.rectangle(box, fill=150)
        # Torn corner: a triangle of older paper.
        if rng.random() < 0.5:
            c = (bx1, by1) if rng.random() < 0.5 else (bx0, by1)
            s = bw * rng.uniform(0.08, 0.18)
            tri = [c, (c[0] + (s if c[0] == bx0 else -s), c[1]), (c[0], c[1] - s * 0.8)]
            draw.polygon(tri, fill=(214, 208, 192))
            rdraw.polygon(tri, fill=120)
    image = image.resize((n, n), Image.LANCZOS)
    relief = relief.resize((n, n), Image.BILINEAR)
    color = np.asarray(image, float)
    wrinkles = _stretched(np.random.default_rng(int(rng.integers(1 << 30))), (n, n), (3, 40)) * 0.6 + _fbm(rng, n, 32, 3) * 0.4
    color *= (0.9 + 0.12 * _fbm(rng, n, 8, 4))[..., None]
    height = np.asarray(relief, float) / 255 + wrinkles * 0.3
    return _u8(color), _normal(_blur(height, 1) * 3, 1.0), pack_orm(1.0, 0.75 + 0.1 * wrinkles, 0.0)


# Vehicles


def _make_carpaint(spec, n, maps, rng):
    """Near-white car paint for vertex colours, glossy like clearcoat: a faint orange peel and a little dust."""
    peel = _fbm(rng, n, 32, 3)
    dust = _fbm(rng, n, 4, 4)
    color = np.asarray(spec["fallback"]["color"], float) * (0.985 + 0.015 * dust)[..., None]
    return _u8(color), _normal(peel, 0.04), pack_orm(1.0, 0.08 + 0.05 * dust, 0.3)


def _make_rubber(spec, n, maps, rng):
    grain = _fbm(rng, n, 64, 3)
    color = np.asarray(spec["fallback"]["color"], float) * (0.9 + 0.2 * grain)[..., None]
    return _u8(color), _normal(grain, 0.8), pack_orm(1.0, 0.82 + 0.1 * grain, 0.0)


def _make_plastic(spec, n, maps, rng):
    grain = _fbm(rng, n, 32, 3)
    color = np.asarray(spec["fallback"]["color"], float) * (0.92 + 0.15 * grain)[..., None]
    return _u8(color), _normal(grain, 0.6), pack_orm(1.0, 0.55 + 0.15 * grain, 0.0)


def _make_vehicledetails(spec, n, maps, rng):
    """2 × 2 vehicle atlas: an EU number plate in the top 22 % of cell 0, a taxi roof sign (cell 1), a dark bus
    destination display with amber dot-matrix text (cell 2), and a honeycomb radiator grille (cell 3)."""
    half = n // 2
    image = Image.new("RGB", (n, n), (40, 40, 40))
    relief = Image.new("L", (n, n), 128)
    metal = Image.new("L", (n, n), 0)
    draw, rdraw = ImageDraw.Draw(image), ImageDraw.Draw(relief)
    # Plate: 520 × 110 mm → full cell width, 22 % of its height.
    ph = int(half * 0.22)
    draw.rectangle((0, 0, half, ph), fill=(240, 240, 236))
    draw.rectangle((0, 0, half, ph), outline=(20, 20, 20), width=3)
    draw.rectangle((3, 3, int(half * 0.085), ph - 3), fill=(0, 51, 153))
    draw.text((half * 0.55, ph / 2), "B AX 1042", fill=(20, 20, 20), font=_font(int(ph * 0.62), bold=False), anchor="mm")
    rdraw.rectangle((0, 0, half, ph), fill=170)
    draw.rectangle((0, ph, half, half), fill=(30, 30, 30))
    # Taxi sign.
    draw.rectangle((half, 0, n, half), fill=(242, 196, 30))
    draw.text((half * 1.5, half * 0.5), "TAXI", fill=(18, 18, 18), font=_font(int(half * 0.32)), anchor="mm")
    # Destination display: dark glass with amber dot matrix.
    display = Image.new("L", (half, half), 0)
    ImageDraw.Draw(display).text((half * 0.5, half * 0.5), "100 Hauptbahnhof", fill=255,
                                 font=_font(int(half * 0.085)), anchor="mm")
    cells = -(-half // 6)
    dots = np.asarray(display.resize((cells, cells), Image.BILINEAR), float) / 255 > 0.35
    yy, xx = np.mgrid[0:half, 0:half]
    dot = ((yy % 6 - 2.5) ** 2 + (xx % 6 - 2.5) ** 2 < 6) & dots[yy // 6, xx // 6]
    panel = np.full((half, half, 3), (16, 16, 18), float)
    panel[dot > 0] = (255, 170, 30)
    image.paste(Image.fromarray(panel.astype(np.uint8)), (0, half))
    # Grille: honeycomb holes.
    gy, gx = np.mgrid[0:half, 0:half]
    cell = 24
    row = gy // (cell * 0.866)
    hx = (gx + (row % 2) * cell / 2) % cell - cell / 2
    hy = gy % (cell * 0.866) - cell * 0.433
    hole = (np.abs(hx) * 0.866 + np.abs(hy) * 0.5 < cell * 0.36) & (np.abs(hy) < cell * 0.36)
    grille = np.where(hole[..., None], (8, 8, 9), (36, 37, 39)).astype(np.uint8)
    image.paste(Image.fromarray(grille), (half, half))
    relief.paste(Image.fromarray(np.where(hole, 40, 200).astype(np.uint8)), (half, half))
    color = np.asarray(image)
    height = np.asarray(relief, float) / 255
    rough = np.full((n, n), 0.5)
    rough[half:, :half] = 0.08  # Display glass.
    return color, _normal(_blur(height, 1) * 4, 1.0), pack_orm(1.0, rough, np.asarray(metal, float) / 255)


# Vegetation


def _make_grass(spec, n, maps, rng):
    """A grass tuft (unit UVs, v up): about 70 tapering, curving blades from the bottom centre; alpha cut-out, darker
    at the base, with per-blade tilted normals."""
    s = 2  # Draw at double size, then downsample for smooth edges.
    m = n * s
    color = Image.new("RGBA", (m, m), (0, 0, 0, 0))
    normal = Image.new("RGB", (m, m), FLAT)
    draw, ndraw = ImageDraw.Draw(color), ImageDraw.Draw(normal)
    base = np.asarray(spec["fallback"]["color"], float)
    for _ in range(70):
        x0 = m / 2 + rng.normal(0, m * 0.08)
        height = m * rng.uniform(0.45, 0.97)
        lean = rng.normal(0, 0.35)
        width = m * rng.uniform(0.012, 0.022)
        t = np.linspace(0, 1, 12)
        xs = x0 + lean * height * t ** 1.6 * 0.6
        ys = m - t * height * (1 - 0.15 * abs(lean) * t)
        left = [(x - width * (1 - u) / 2, y) for x, y, u in zip(xs, ys, t)]
        right = [(x + width * (1 - u) / 2, y) for x, y, u in zip(xs, ys, t)][::-1]
        hue = base * rng.uniform(0.75, 1.25) * np.array([rng.uniform(0.9, 1.15), 1, rng.uniform(0.8, 1.0)])
        tilt = np.array([lean * 0.6 + rng.normal(0, 0.2), rng.normal(0, 0.15), 1.0])
        tilt /= np.linalg.norm(tilt)
        draw.polygon(left + right, fill=tuple(int(v) for v in np.clip(hue, 0, 255)) + (255,))
        ndraw.polygon(left + right, fill=tuple(int(v) for v in (tilt + 1) * 127.5))
    color = np.asarray(color.resize((n, n), Image.LANCZOS), float)
    normal = np.asarray(normal.resize((n, n), Image.LANCZOS))
    v = np.linspace(0, 1, n)[:, None]  # 0 at the top row, 1 at the bottom.
    color[..., :3] *= (1.05 - 0.55 * v ** 2)[..., None]
    alpha = (color[..., 3] > 110).astype(float)
    occlusion = 1 - 0.55 * v ** 1.5 * np.ones((1, n))
    return _rgba(color[..., :3], alpha), normal, pack_orm(occlusion, 0.75, 0.0)


FLOWER_COLOURS = [((214, 40, 40), (250, 210, 40)), ((250, 214, 40), (230, 120, 20)), ((246, 246, 240), (250, 200, 40)),
                  ((150, 60, 190), (250, 230, 120))]


def _make_flowers(spec, n, maps, rng):
    """2 × 2 atlas of flower clumps (v up): leafy stems with five-petal blossoms, one colour scheme per cell
    (red, yellow, white, purple)."""
    half = n // 2
    color = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    normal = Image.new("RGB", (n, n), FLAT)
    leaf = np.asarray(spec["fallback"]["color"], float)
    for cell, (petal, centre) in enumerate(FLOWER_COLOURS):
        cc, cn = Image.new("RGBA", (half, half), (0, 0, 0, 0)), Image.new("RGB", (half, half), FLAT)
        draw, ndraw = ImageDraw.Draw(cc), ImageDraw.Draw(cn)
        for _ in range(18):  # Leaves.
            x0 = half / 2 + rng.normal(0, half * 0.12)
            a = -math.pi / 2 + rng.normal(0, 0.6)
            length = half * rng.uniform(0.15, 0.35)
            tip = (x0 + math.cos(a) * length, half + math.sin(a) * length)
            side = np.array([-math.sin(a), math.cos(a)]) * length * 0.18
            mid = np.array([(x0 + tip[0]) / 2, (half + tip[1]) / 2])
            shade = tuple(int(v) for v in np.clip(leaf * rng.uniform(0.7, 1.2), 0, 255)) + (255,)
            poly = [(x0, half), tuple(mid + side), tip, tuple(mid - side)]
            draw.polygon(poly, fill=shade)
            ndraw.polygon(poly, fill=tuple(int(v) for v in (np.array([math.cos(a) * 0.4, 0.2, 1]) / 1.1 + 1) * 127.5))
        for _ in range(9):  # Stems and blossoms.
            x0 = half / 2 + rng.normal(0, half * 0.15)
            top = (x0 + rng.normal(0, half * 0.08), half * rng.uniform(0.12, 0.5))
            draw.line([(x0, half), top], fill=(70, 110, 50, 255), width=max(2, half // 120))
            r = half * rng.uniform(0.05, 0.08)
            shade = np.asarray(petal, float) * rng.uniform(0.85, 1.1)
            for p in range(5):
                a = p * 2 * math.pi / 5 + rng.uniform(0, 1)
                c = (top[0] + math.cos(a) * r * 0.6, top[1] + math.sin(a) * r * 0.6)
                draw.ellipse((c[0] - r * 0.55, c[1] - r * 0.55, c[0] + r * 0.55, c[1] + r * 0.55),
                             fill=tuple(int(v) for v in np.clip(shade, 0, 255)) + (255,))
            draw.ellipse((top[0] - r * 0.3, top[1] - r * 0.3, top[0] + r * 0.3, top[1] + r * 0.3), fill=centre + (255,))
        corner = (cell % 2 * half, cell // 2 * half)
        color.paste(cc, corner)
        normal.paste(cn, corner)
    rgba = np.asarray(color, float)
    return _rgba(rgba[..., :3], rgba[..., 3] > 128), np.asarray(normal), pack_orm(1.0, 0.6, 0.0)


# Decals: unit UVs, alpha fading out well before the edges.


def _edge_fade(n, margin=0.06):
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    border = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    return np.clip(border / margin, 0, 1)


def _make_manhole(spec, n, maps, rng):
    """A round 62.5 cm cast-iron manhole cover in its frame, set in a ring of darker repair asphalt: raised studs
    polished by traffic, rust in the recesses, two pick holes."""
    r, angle = _radial(n)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    cover, frame, ring = 0.347, 0.395, 0.47
    worn = _fbm(rng, n, 16, 4)
    grid = 0.032
    gx, gy = ((x - 0.5) / grid) % 1, ((y - 0.5) / grid) % 1
    stud = (np.abs(gx - 0.5) < 0.3) & (np.abs(gy - 0.5) < 0.3) & (r < cover - 0.03) & (r > 0.07)
    rim = (r > cover - 0.025) & (r < cover)
    holes = np.zeros((n, n), bool)
    for a in (0.0, math.pi):
        holes |= np.hypot(x - 0.5 - math.cos(a) * 0.25, y - 0.5 - math.sin(a) * 0.25) < 0.012
    raised = stud | rim | ((r < 0.07) & (r > 0.06))
    rust = np.clip(_fbm(rng, n, 8, 4) * 1.8 - 0.8, 0, 1) * ~raised
    iron = np.array([50, 48, 46.0]) * (0.85 + 0.3 * worn[..., None])
    iron = iron * (1 - 0.5 * rust[..., None]) + np.array([96, 62, 38.0]) * 0.5 * rust[..., None]
    iron = np.where(raised[..., None], np.array([104, 102, 98.0]) * (0.85 + 0.3 * worn[..., None]), iron)
    iron[holes] = (10, 10, 10)
    frame_color = np.array([58, 56, 54.0]) * (0.85 + 0.3 * worn[..., None])
    asphalt = np.array([46, 46, 48.0]) * (0.8 + 0.4 * _fbm(rng, n, 64, 2)[..., None])
    gap = (r > cover) & (r < cover + 0.006)
    color = np.where((r < cover)[..., None], iron, np.where((r < frame)[..., None], frame_color, asphalt))
    color[gap] = (14, 14, 14)
    wobble = ring + 0.02 * (_noise(rng, n, 6) - 0.5)
    alpha = np.where(r < frame, 1.0, np.clip((wobble - r) / 0.03, 0, 1) * 0.92)
    height = np.where(r < frame, 0.6 + 0.4 * raised - 0.6 * holes - 0.5 * gap, 0.5 + 0.05 * _fbm(rng, n, 32, 3))
    rough = np.where(raised, 0.35, np.where(r < frame, 0.7, 0.85)) + 0.1 * rust
    metal = np.where(r < frame, np.where(raised, 0.7, 0.3) * (1 - rust), 0.0)
    return _rgba(color, alpha), _normal(_blur(height, 1) * 8, 1.0), pack_orm(0.7 + 0.3 * _blur(height, 3), rough, metal), height


def _make_decalphoto(spec, n, maps, rng):
    """A photographed cut-out (ambientCG with opacity), such as a street drain; procedural grate offline."""
    if maps and "Opacity" in maps:
        color = np.asarray(_resize(maps["Color"], n, "RGB"), float)
        alpha = np.asarray(_resize(maps["Opacity"], n, "L"), float) / 255 * _edge_fade(n, 0.02)
        normal = np.asarray(_resize(maps["NormalGL"], n, "RGB"))
        gray = lambda key, fill: np.asarray(_resize(maps[key], n, "L"), float) / 255 if key in maps else np.full((n, n), fill)
        height = gray("Displacement", 0.5)
        return _rgba(color, alpha), normal, pack_orm(gray("AmbientOcclusion", 1.0), gray("Roughness", 0.7),
                                                     gray("Metalness", 0.0)), height
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    frame = (np.abs(x - 0.5) < 0.36) & (np.abs(y - 0.5) < 0.3)
    bars = frame & (np.abs(x - 0.5) < 0.3) & (np.abs(y - 0.5) < 0.24) & (((x - 0.2) / 0.06) % 1 > 0.45)
    worn = _fbm(rng, n, 16, 4)
    color = np.where(frame[..., None], np.array([60, 56, 52.0]) * (0.8 + 0.4 * worn[..., None]), np.array([128, 126, 120.0]))
    color[bars] = (12, 12, 12)
    alpha = np.where(frame, 1.0, np.clip(1 - (np.maximum(np.abs(x - 0.5) - 0.36, np.abs(y - 0.5) - 0.3)) / 0.08, 0, 1) * 0.9)
    height = frame * 1.0 - bars * 1.0
    return _rgba(color, alpha), _normal(_blur(height, 1) * 6, 1.0), pack_orm(1 - 0.6 * bars, 0.6, 0.3 * frame), height


def _make_crack(spec, n, maps, rng):
    """Branching asphalt or concrete cracks: dark grooves with a faint darker halo, fading out toward the edges."""
    s = 2
    m = n * s
    lines = Image.new("L", (m, m), 0)
    halo = Image.new("L", (m, m), 0)
    draw, hdraw = ImageDraw.Draw(lines), ImageDraw.Draw(halo)

    def crack(p, a, length, width, depth):
        points = [p]
        for _ in range(int(length / (m * 0.012))):
            a += rng.normal(0, 0.16)
            p = (p[0] + math.cos(a) * m * 0.012, p[1] + math.sin(a) * m * 0.012)
            points.append(p)
            if depth < 2 and rng.random() < 0.07:
                crack(p, a + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), length * 0.45, width * 0.6, depth + 1)
        for (p0, p1), t in zip(zip(points, points[1:]), np.linspace(1, 0.3, len(points))):
            draw.line([p0, p1], fill=255, width=max(1, int(width * t)))
            hdraw.line([p0, p1], fill=110, width=max(2, int(width * t * 5)))

    for _ in range(2):
        start = (m * rng.uniform(0.35, 0.65), m * rng.uniform(0.35, 0.65))
        a = rng.uniform(0, 2 * math.pi)
        crack(start, a, m * 0.45, m * 0.009, 0)
        crack(start, a + math.pi + rng.normal(0, 0.3), m * 0.4, m * 0.009, 0)
    groove = np.asarray(lines.resize((n, n), Image.LANCZOS), float) / 255
    halo = _blur(np.asarray(halo.resize((n, n), Image.BILINEAR), float) / 255, 3)
    fade = _edge_fade(n, 0.2)
    alpha = np.clip(groove * 0.95 + halo * 0.5, 0, 1) * fade
    color = np.array([22, 22, 22.0]) * (1 - groove[..., None]) + np.array([8, 8, 8.0]) * groove[..., None]
    height = -groove - halo * 0.2
    return _rgba(color, alpha), _normal(_blur(height, 1) * 6, 1.0), pack_orm(1 - 0.5 * groove, 0.8, 0.0), height


def _make_patch(spec, n, maps, rng):
    """An asphalt repair patch: darker, smoother fresh asphalt in an irregular rectangle with a glossy bitumen seam."""
    base = _surface(dict(spec, tint=spec["fallback"]["color"]), n, maps, rng) if maps else None
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    wobble = (_noise(rng, n, 5) - 0.5) * 0.05 + (_noise(rng, n, 24) - 0.5) * 0.01
    box = np.maximum(np.abs(x - 0.5) / 0.42, np.abs(y - 0.5) / 0.4)
    inside = 0.42 * (1 - box) + wobble  # > 0 inside the patch, metres-ish distance to the edge in UV.
    seam = np.clip(1 - np.abs(inside) / 0.008, 0, 1)
    alpha = np.clip(inside / 0.004 + 0.5, 0, 1)
    alpha = np.maximum(alpha, seam)
    grain = _fbm(rng, n, 64, 3)
    color = base["color"] if base is not None else np.asarray(spec["fallback"]["color"], float) * (0.85 + 0.3 * grain)[..., None]
    color = color * 0.85
    color = color * (1 - seam[..., None]) + np.array([14, 14, 15.0]) * seam[..., None]
    rough = (base["rough"] * 0.9 if base is not None else 0.8 + 0.1 * grain) * (1 - 0.5 * seam)
    normal = base["normal"] if base is not None else np.dstack((np.zeros((n, n, 2)), np.ones((n, n))))
    height = (inside > 0) * 0.3 + seam * 0.2 + grain * 0.05
    return _rgba(color, alpha), _detail(normal, seam * 3, 1.0), pack_orm(1.0, rough, 0.0), height


def _make_stain(spec, n, maps, rng):
    """An oil stain: overlapping translucent blobs with darker cores and rings, glossier than the road."""
    r, _ = _radial(n)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    density = np.zeros((n, n))
    for _ in range(6):
        cx, cy = 0.5 + rng.normal(0, 0.09, 2)
        rad = rng.uniform(0.08, 0.2)
        d = np.hypot(x - cx, (y - cy) * rng.uniform(0.7, 1.3)) / rad
        density += np.clip(1 - d, 0, 1) ** 0.7 * rng.uniform(0.4, 0.9)
    warp = _fbm(rng, n, 6, 5)
    density = np.clip(density * (0.6 + 0.8 * warp) - 0.15, 0, 1) * _edge_fade(n, 0.15)
    ring = np.clip(1 - np.abs(density - 0.15) / 0.06, 0, 1) * 0.3
    alpha = np.clip(density * 0.75 + ring, 0, 0.85)
    color = np.asarray(spec["fallback"]["color"], float) * np.ones((n, n, 3))
    color *= (1 + 0.6 * _fbm(rng, n, 32, 2))[..., None]
    return _rgba(color, alpha), np.full((n, n, 3), FLAT, np.uint8), pack_orm(1.0, 0.35 - 0.2 * density, 0.0)


def _make_streak(spec, n, maps, rng):
    """Vertical dirt streaks running down a wall from a sill or cornice (unit UVs, 1:3, v up): a soft wash under the
    top edge, then tapered streaks and thin drips of uneven length."""
    rows, cols = n * 3 // 2, n // 2
    t = np.linspace(0, 1, rows)[:, None]  # 0 at the top row.
    x = (np.arange(cols)[None, :] + 0.5) / cols
    fine = _stretched(rng, (rows, cols), (2, 64), 4)
    alpha = 0.45 * np.clip(1 - t / 0.12, 0, 1) ** 1.5 * np.clip(1 - np.abs(x - 0.5) / 0.47, 0, 1) ** 0.5
    for _ in range(14):
        cx, width = rng.uniform(0.08, 0.92), rng.uniform(0.015, 0.09)
        start, length = rng.uniform(0, 0.08), rng.uniform(0.25, 0.95)
        u = np.clip((t - start) / length, 0, 1)
        along = np.where(t >= start, (1 - u) ** 1.4, 0) * rng.uniform(0.35, 0.8)
        wander = cx + 0.01 * np.sin(t * rng.uniform(4, 9) + rng.uniform(0, 6))
        across = np.clip(1 - np.abs(x - wander) / (width * (1 - 0.5 * u)), 0, 1) ** 0.8
        alpha = alpha + along * across
    for _ in range(10):  # Thin drips.
        cx, length = rng.uniform(0.05, 0.95), rng.uniform(0.3, 1.0)
        across = np.clip(1 - np.abs(x - cx) / 0.006, 0, 1)
        alpha = alpha + across * np.clip(1 - t / length, 0, 1) * 0.5
    alpha = np.clip(alpha * (0.7 + 0.5 * fine), 0, 0.85) * np.clip(np.minimum(x, 1 - x) / 0.04, 0, 1)
    alpha *= np.clip(t / 0.01, 0, 1) * np.clip((1 - t) / 0.05, 0, 1)
    color = np.asarray(spec["fallback"]["color"], float) * (0.7 + 0.5 * fine[..., None])
    return (_rgba(color, alpha), np.full((rows, cols, 3), FLAT, np.uint8),
            pack_orm(np.ones((rows, cols)), 0.85 + 0.1 * alpha, 0.0))


def _make_grime(spec, n, maps, rng):
    """Ground-contact grime for wall bottoms (unit UVs, 4:1, v up, tiles along u): dark at the ground, fading upwards
    in a ragged edge, with splash spots."""
    rows, cols = n // 4, n
    noise = _fbm(rng, n, 8, 5)[:rows]  # Rows of tileable noise stay tileable along u.
    fine = _fbm(rng, n, 64, 3)[:rows]
    v = np.linspace(1, 0, rows)[:, None]  # 0 at the bottom row (the ground), 1 at the top.
    edge = 0.45 + 0.35 * noise
    alpha = np.clip((edge - v) / 0.25, 0, 1) ** 1.3 * (0.65 + 0.35 * fine)
    splash = (fine > 0.72) & (v < 0.35)
    alpha = np.clip(alpha + splash * 0.3, 0, 0.9)
    color = np.asarray(spec["fallback"]["color"], float) * (0.75 + 0.5 * fine[..., None])
    return (_rgba(color, alpha), np.full((rows, cols, 3), FLAT, np.uint8),
            pack_orm(1 - 0.3 * alpha, 0.9, 0.0))


def _make_decalleaves(spec, n, maps, rng):
    """Fallen leaves in a loose drift (alpha cut-out): linden- and maple-like leaves in autumn colours, denser in the
    middle, each with a midrib and a slight curl in the normal map."""
    s = 2
    m = n * s
    color = Image.new("RGBA", (m, m), (0, 0, 0, 0))
    normal = Image.new("RGB", (m, m), FLAT)
    draw, ndraw = ImageDraw.Draw(color), ImageDraw.Draw(normal)
    palette = np.array([(190, 140, 40), (170, 90, 30), (120, 70, 35), (200, 170, 60), (150, 120, 50), (110, 120, 40)], float)
    for _ in range(260):
        r = abs(rng.normal(0, 0.17))
        if r > 0.42:
            continue
        a = rng.uniform(0, 2 * math.pi)
        c = np.array([0.5 + math.cos(a) * r, 0.5 + math.sin(a) * r]) * m
        size = m * rng.uniform(0.025, 0.045)
        turn = rng.uniform(0, 2 * math.pi)
        lobes = rng.integers(0, 2)
        t = np.linspace(0, 2 * math.pi, 28, endpoint=False)
        radius = size * (0.6 + 0.4 * np.abs(np.sin(t / 2))) * (1 + (0.25 * np.cos(5 * t) if lobes else 0.05 * np.cos(14 * t)))
        radius *= np.where(np.cos(t) > 0.95, 1.3, 1.0)  # Pointed tip.
        pts = [(c[0] + math.cos(tt + turn) * rr, c[1] + math.sin(tt + turn) * rr * 0.8) for tt, rr in zip(t, radius)]
        base = palette[rng.integers(len(palette))] * rng.uniform(0.8, 1.15)
        fill = tuple(int(v) for v in np.clip(base, 0, 255)) + (255,)
        draw.polygon(pts, fill=fill)
        tip = (c[0] + math.cos(turn) * size * 1.2, c[1] + math.sin(turn) * size * 0.96)
        stem = (c[0] - math.cos(turn) * size * 1.3, c[1] - math.sin(turn) * size * 1.04)
        draw.line([stem, tip], fill=tuple(int(v * 0.7) for v in fill[:3]) + (255,), width=max(1, m // 600))
        tilt = rng.normal(0, 0.35, 2)
        ndraw.polygon(pts, fill=tuple(int(v) for v in (np.array([tilt[0], tilt[1], 1]) / np.linalg.norm([*tilt, 1]) + 1) * 127.5))
    rgba = np.asarray(color.resize((n, n), Image.LANCZOS), float)
    normal = np.asarray(normal.resize((n, n), Image.LANCZOS))
    alpha = rgba[..., 3] > 128
    rgba[..., :3] *= (0.85 + 0.25 * _fbm(rng, n, 32, 3))[..., None]
    return _rgba(rgba[..., :3], alpha), normal, pack_orm(1.0, 0.7, 0.0)


MAKERS = dict(slabs=_make_slabs, dressed=_make_dressed, seams=_make_seams, treegrate=_make_treegrate, awning=_make_awning,
              shopsigns=_make_shopsigns, posters=_make_posters, carpaint=_make_carpaint, rubber=_make_rubber,
              plastic=_make_plastic, vehicledetails=_make_vehicledetails, grass=_make_grass, flowers=_make_flowers,
              manhole=_make_manhole, decalphoto=_make_decalphoto, crack=_make_crack, patch=_make_patch,
              stain=_make_stain, streak=_make_streak, grime=_make_grime, decalleaves=_make_decalleaves)
