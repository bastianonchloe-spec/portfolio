#!/usr/bin/env python3
"""Génère les pages servies par adresse à partir de index.html.

index.html reste la seule source : ce script en dérive, au moment du
déploiement, une copie par adresse afin que chaque adresse réponde 200
(au lieu de 404) et soit correctement indexée et partagée.

- 404.html : fichier de secours pour les adresses inconnues
- un dossier par adresse : /artists/index.html, /artists/neen/index.html, ...

Chaque copie reçoit <base href="/"> pour que les images (assets/...)
soient résolues depuis la racine du site.

Lancé automatiquement par .github/workflows/pages.yml.
"""

import io
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADRESSES = [
    "artists",
    "blog",
    "contact",
    "artists/neen",
    "artists/eliott-ray",
    "artists/suze",
    "artists/baala",
    "blog/pollen",
]


def main():
    source = os.path.join(RACINE, "index.html")
    page = io.open(source, encoding="utf-8").read()

    copie = re.sub(r"(<head>)", r'\1\n    <base href="/">', page, count=1)
    if copie == page:
        sys.exit("balise <head> introuvable dans index.html")

    ecrites = []

    io.open(os.path.join(RACINE, "404.html"), "w", encoding="utf-8").write(copie)
    ecrites.append("404.html")

    for adresse in ADRESSES:
        dossier = os.path.join(RACINE, *adresse.split("/"))
        os.makedirs(dossier, exist_ok=True)
        io.open(os.path.join(dossier, "index.html"), "w", encoding="utf-8").write(copie)
        ecrites.append(adresse + "/index.html")

    print("%d pages générées : %s" % (len(ecrites), ", ".join(ecrites)))


if __name__ == "__main__":
    main()
