import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.config.config_loader import ConfigLoader
from src.http.routes.routeable import routeable
from src import utils

logger = utils.get_logger()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_-]+\.(wav|lip)$")


class audio_route(routeable):
    """GET /audio/{voice}/{filename} — serves voicelines from the local
    Mantella mod_path so a remote Skyrim mod (e.g. running on Android via
    GameHub Wine) can fetch them over HTTP and write them to its own
    Data/Sound/Voice/Mantella.esp/<voice>/ folder before Topic.Say plays them.
    """

    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)

    def _setup_route(self):
        pass

    def _can_route_be_used(self) -> bool:
        return True

    def add_route_to_server(self, app: FastAPI):
        @app.get("/audio/{voice}/{filename}")
        async def get_audio(voice: str, filename: str):
            if not _SAFE_NAME.match(voice):
                raise HTTPException(status_code=400, detail="invalid voice")
            if not _SAFE_FILENAME.match(filename):
                raise HTTPException(status_code=400, detail="invalid filename")

            mod_path = self._config.mod_path
            target = (Path(mod_path) / voice / filename).resolve()
            mod_root = Path(mod_path).resolve()

            try:
                target.relative_to(mod_root)
            except ValueError:
                raise HTTPException(status_code=400, detail="path escapes mod folder")

            if not target.is_file():
                raise HTTPException(status_code=404, detail="not found")

            return FileResponse(
                str(target),
                media_type="application/octet-stream",
                filename=filename,
            )
