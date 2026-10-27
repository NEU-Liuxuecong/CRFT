"""Path resolution for the Kechaba delivery tree.

Every entry point locates the project root from its own __file__, so the working
directory of the caller never matters.
"""
import json
from pathlib import Path

KROOT = Path(__file__).resolve().parents[1]          # .../Kechaba
SRC = KROOT / "src"
RUNTIME = SRC / "runtime"
F13DIR = SRC / "f13"
BRIDGEDIR = SRC / "bridge"
CONFIGS = KROOT / "configs"
WEIGHTS = KROOT / "weights"
OUTPUTS = KROOT / "outputs"
RUNS = KROOT / "runs"


def machine_paths():
    """Load machine paths and resolve relative paths from project root."""
    cfg = json.loads(
        (CONFIGS / "machine_paths.json").read_text(encoding="utf-8")
    )

    def resolve(value):
        if not value:
            return value
        p = Path(value).expanduser()
        return str((KROOT / p).resolve()) if not p.is_absolute() else str(p)

    for key in (
        "crft_root",
        "crft_ckpt_os",
        "crft_ckpt_roadscene",
        "d1_os",
        "d1_roadscene",
        "data_root_os",
        "data_root_roadscene",
    ):
        cfg[key] = resolve(cfg.get(key, ""))

    for dataset, splits in cfg.get("cache_dirs", {}).items():
        for split, value in splits.items():
            splits[split] = resolve(value)

    return cfg


def ensure_sys_path():
    """Put the D1 runtime and the F13 folder on sys.path exactly once."""
    import sys
    for p in (str(RUNTIME), str(F13DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def describe_modules():
    """Report the real __file__ of every key module (used by --check-config)."""
    import sys
    mods = {}
    for name in ("geo_core_v1", "global_rescue", "diverse_rescue", "geometry_prior512", "flow_bridge"):
        m = sys.modules.get(name)
        mods[name] = str(Path(m.__file__).resolve()) if m is not None and getattr(m, "__file__", None) else None
    return mods
