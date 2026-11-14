#!/usr/bin/env python3
"""Extrait les etoiles des visuels d'accueil (home-1.png, home-2.png) en
calques PNG transparents, puis ecrit le bloc HTML qui les anime.

Chaque etoile devient une image independante, superposee exactement sur sa
position d'origine dans le visuel. A l'echelle 1 le calque recouvre l'etoile
d'origine ; l'animation CSS le fait ensuite grandir et retrecir.
"""

import io
import os
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER = os.path.join(RACINE, "assets", "stars")
PINK = np.array([255, 106, 148], dtype=np.float32)
SEED = 20261114


def components(mask, min_area=200, max_side=220):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = (int(stats[i, k]) for k in (0, 1, 2, 3, 4))
        if area < min_area or bw > max_side or bh > max_side:
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        out.append({"label": i, "x": x, "y": y, "w": bw, "h": bh, "area": area})
    out.sort(key=lambda c: (c["y"], c["x"]))
    return labels, out


def save_sprite(rgba, crop, path):
    x, y, cw, ch = crop
    Image.fromarray(rgba[y:y + ch, x:x + cw]).save(path, optimize=True)


def extract_pink(img, prefix):
    """home-1 : etoiles rose uni sur une photo."""
    rgb = img[:, :, :3].astype(np.float32)
    dist = np.abs(rgb - PINK).max(axis=2)
    mask = dist < 60
    labels, comps = components(mask)
    sprites = []
    for idx, c in enumerate(comps, 1):
        sel = labels == c["label"]
        pad = int(round(max(c["w"], c["h"]) * 0.28))
        x, y = c["x"] - pad, c["y"] - pad
        cw, ch = c["w"] + 2 * pad, c["h"] + 2 * pad
        rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
        rgba[pad:pad + c["h"], pad:pad + c["w"], :3] = PINK.astype(np.uint8)
        alpha = (sel[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] * 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
        rgba[pad:pad + c["h"], pad:pad + c["w"], 3] = alpha
        name = f"{prefix}-{idx:02d}.png"
        save_sprite(rgba, (0, 0, cw, ch), os.path.join(DOSSIER, name))
        sprites.append({"file": name, "x": x, "y": y, "w": cw, "h": ch})
    return sprites


def extract_cutout(img, prefix):
    """home-2 : etoiles en collage sur fond rose uni -> alpha demelange."""
    rgb = img[:, :, :3].astype(np.float32)
    # P = a*S + (1-a)*rose ; sur le canal rouge-vert : P_R-P_G = (1-a)*149
    alpha = 1.0 - (rgb[:, :, 0] - rgb[:, :, 1]) / 149.0
    alpha = np.clip(alpha, 0.0, 1.0)
    mask = alpha > 0.35
    labels, comps = components(mask, min_area=200)
    sprites = []
    for idx, c in enumerate(comps, 1):
        sel = labels == c["label"]
        pad = int(round(max(c["w"], c["h"]) * 0.28))
        alpha_c = alpha[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]].copy()
        alpha_c[~sel[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]] = 0.0
        # demelange : S = (P - (1-a)*rose) / a
        sub = rgb[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        a = np.clip(alpha_c, 1e-3, 1.0)[:, :, None]
        color = (sub - (1.0 - a) * PINK) / a
        color = np.clip(color, 0, 255)
        color = np.where(alpha_c[:, :, None] > 0.05, color, 0)
        rgba = np.zeros((c["h"], c["w"], 4), dtype=np.uint8)
        rgba[:, :, :3] = color.astype(np.uint8)
        rgba[:, :, 3] = (np.clip(alpha_c, 0, 1) * 255).astype(np.uint8)
        canvas = np.zeros((c["h"] + 2 * pad, c["w"] + 2 * pad, 4), dtype=np.uint8)
        canvas[pad:pad + c["h"], pad:pad + c["w"]] = rgba
        name = f"{prefix}-{idx:02d}.png"
        Image.fromarray(canvas).save(os.path.join(DOSSIER, name), optimize=True)
        sprites.append({"file": name, "x": c["x"] - pad, "y": c["y"] - pad,
                        "w": c["w"] + 2 * pad, "h": c["h"] + 2 * pad})
    return sprites


def html_block(sprites, rng):
    lignes = ['<div class="star-layer" aria-hidden="true">', '  <div class="star-stage">']
    for s in sprites:
        petit = s["w"] < 90
        if petit:
            dur = rng.uniform(1.8, 2.9)
            amp = rng.uniform(1.20, 1.32)
        else:
            dur = rng.uniform(2.2, 3.4)
            amp = rng.uniform(1.14, 1.24)
        delay = -rng.uniform(0, dur)
        style = ("left:%.3f%%;top:%.3f%%;width:%.3f%%;--star-dur:%.2fs;"
                 "--star-delay:%.2fs;--star-amp:%.3f") % (
            s["x"] / 1264 * 100, s["y"] / 974 * 100, s["w"] / 1264 * 100, dur, delay, amp)
        lignes.append(
            '    <img class="star" src="assets/stars/%s" alt="" width="%d" height="%d" style="%s">'
            % (s["file"], s["w"], s["h"], style)
        )
    lignes += ["  </div>", "</div>"]
    return "\n".join(lignes)


def main():
    os.makedirs(DOSSIER, exist_ok=True)
    rng = random.Random(SEED)
    pages = {}
    for name, extract in (("home-1", extract_pink), ("home-2", extract_cutout)):
        img = np.array(Image.open(os.path.join(RACINE, "assets", name + ".png")).convert("RGBA"))
        sprites = extract(img, name)
        pages[name] = html_block(sprites, rng)
        print(name, len(sprites), "etoiles")

    chemin = os.path.join(RACINE, "index.html")
    page = io.open(chemin, encoding="utf-8").read()
    for name, bloc in pages.items():
        motif = re.compile(r"<!-- stars:%s -->.*?<!-- /stars:%s -->" % (name, name), re.S)
        if not motif.search(page):
            sys.exit("reperes stars:%s introuvables dans index.html" % name)
        indent = "\n".join("          " + ligne for ligne in bloc.split("\n"))
        page = motif.sub(
            lambda _m: "<!-- stars:%s -->%s<!-- /stars:%s -->" % (name, indent, name), page)
    io.open(chemin, "w", encoding="utf-8").write(page)
    print("calques HTML ecrits dans index.html")


if __name__ == "__main__":
    main()
