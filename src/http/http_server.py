import logging
import click
from fastapi import FastAPI
import uvicorn
from src.http.routes.routeable import routeable
from src import utils

logger = utils.get_logger()


class http_server:
    """A simple http server using FastAPI. Can be started using different routes.
    """
    def __init__(self) -> None:
        self.__app = FastAPI()

        ### Deactivate the logging to console by FastAPI
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        def secho(text, file=None, nl=None, err=None, color=None, **styles):
            pass

        def echo(text, file=None, nl=None, err=None, color=None, **styles):
            pass

        click.echo = echo
        click.secho = secho
        ### End of deactivate logging

    @property
    def app(self) -> FastAPI:
        return self.__app

    def _setup_routes(self, routes: list[routeable]):
        """Sets up the provided server routes
        
        Args:
            routes (list[routeable]): The list of routes to set up
        """
        for route in routes:
            route.add_route_to_server(self.__app)
        return self.__app

    def start(self, port: int, routes: list[routeable], play_startup_sound: bool, show_debug: bool = False):
        """Starts the server and sets up the provided routes

        Args:
            routes (list[routeable]): The list of routes to start
            show_debug (bool, optional): should debug output be shown
        """
        self._setup_routes(routes)

        if play_startup_sound:
            utils.play_mantella_ready_sound()
        
        logger.log(24, '\nConversations not starting when you select an NPC? See here:')
        logger.log(25, 'https://art-from-the-machine.github.io/Mantella/pages/issues_qna')
        logger.log(24, '\nWaiting for player to select an NPC...')
    
        # Bind on 0.0.0.0 so the mod can reach the server over LAN
        # (e.g. Skyrim running on a different machine — Android handheld via
        # GameHub Wine — pointing its HttpHost at the server's LAN IP).
        # 127.0.0.1-only installs still work because 0.0.0.0 includes loopback.
        # access_log=True so every HTTP request from the mod shows up — vital
        # for diagnosing mod<->server issues when running on separate machines.
        uvicorn.run(self.__app, host="0.0.0.0", port=port, access_log=True, log_level="info")
