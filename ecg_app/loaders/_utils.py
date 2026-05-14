from pathlib import Path


def find_companion_file(base_path, extensions):
    base = Path(base_path)
    folder = base.parent
    stem = base.name

    candidates = []
    for ext in extensions:
        candidates.extend([
            folder / f"{stem}{ext.lower()}",
            folder / f"{stem}{ext.upper()}",
            folder / f"{stem}{ext.capitalize()}",
        ])

    for path in candidates:
        if path.exists():
            return str(path)

    return None
