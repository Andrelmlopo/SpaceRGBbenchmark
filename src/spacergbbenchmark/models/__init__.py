from importlib import import_module
from importlib.metadata import entry_points

BUILTINS = {
    "fixture": ("fixture", "FixtureModel"),
    "picopose": ("native", "PicoPoseModel"),
    "pico_hat": ("native", "PicoHatModel"),
    "megapose": ("native", "MegaPoseModel"),
    "mega_hat": ("native", "MegaHatModel"),
    "gigapose_refined": ("native", "GigaPoseModel"),
    "rgbtrack": ("native", "RGBTrackModel"),
    "srt3d": ("srt3d", "SRT3DModel"),
}


def model_class(name):
    if name in BUILTINS:
        module, cls = BUILTINS[name]
        return getattr(import_module(f"spacergbbenchmark.models.{module}"), cls)
    plugins = [p for p in entry_points(group="spacergbbenchmark.models") if p.name == name]
    if len(plugins) != 1:
        raise ValueError(
            f"Unknown or ambiguous model {name!r}; installed models: {sorted(BUILTINS)}"
        )
    return plugins[0].load()
