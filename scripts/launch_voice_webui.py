from __future__ import annotations

# The implementation module keeps its historical name for compatibility with the
# original Slice 14 launcher. Voice-specific behavior now comes from the registry.
from launch_fuxuan_webui import run


if __name__ == "__main__":
    raise SystemExit(run())
