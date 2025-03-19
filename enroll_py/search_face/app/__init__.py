import logging
import logging.config
import time
from importlib import import_module

import yaml
from flask import Flask, request, g
from flask_cors import CORS

import app.utils
import modules
from modules.driver import elasticsearch_driver
from modules.driver.es_logger import EsLogger


LOGGER = logging.getLogger('api')
ES_LOGGER: EsLogger = None


def register_extensions(flask_app, config):
    LOGGER.info("=================== CONFIG EXTENSIONS =======================")

    CORS(flask_app)

    if config.monitor:
        import os
        try:
            os.mkdir("metrics")
        except:
            LOGGER.info("Folder 'metrics' is exists.")
            raise
        os.environ['prometheus_multiproc_dir'] = "metrics"
        from prometheus_flask_exporter.multiprocess import UWsgiPrometheusMetrics
        # Add prometheus wsgi middleware to route /metrics requests
        metrics = UWsgiPrometheusMetrics(flask_app)
        metrics.register_endpoint('/metrics')

    @flask_app.route('/health_check')
    def health_check():
        return str(0)


def register_blueprints(flask_app, config):
    LOGGER.info("=================== CONFIG BLUEPRINTS =======================")
    for module_name in config.apis:
        module = import_module('app.api.{}.routes'.format(module_name))
        flask_app.register_blueprint(module.blueprint)


def configure_logs(config):
    LOGGER.info("=================== CONFIG LOGGER =======================")
    with open(config.log_config_file, 'r') as f:
        log_cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(log_cfg)


def configure_api(flask_app, config):
    LOGGER.info("====================== CONFIG API ===========================")
    modules.init_module(config)

    global ES_LOGGER
    ES_LOGGER = EsLogger(config)

    @flask_app.before_request
    def start_timer():
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        app_id = request.headers.get("app-id", None)
        host = request.host.split(':', 1)[0]
        if app_id is None:
            app_id = request.headers.get("app_id", None)
        params = dict(request.args)

        line = f"<<== method={request.method} path={request.path} ip={ip_address} host={host} app_id={app_id} params={params}"

        LOGGER.info(line)
        g.start = time.time()

    @flask_app.after_request
    def log_request(response):
        if (
                request.path == "/favicon.ico"
                or request.path == "/metrics"
                or request.path.startswith("/static")
                or request.path.startswith("/admin/static")
        ):
            return response
        now = time.time()
        duration = now - g.start
        line = f"==>> status={response.status_code}  duration={duration}s  response={response.json}"
        LOGGER.info(line)
        input_body = request.data.decode("utf8")
        app_id = request.headers.get("app-id", None)
        if app_id is None:
            app_id = request.headers.get("app_id", None)
        if len(input_body) > 5000:
            input_body = None
        if config.enable_log:
            ES_LOGGER.log_request(request.path,
                                  request.method,
                                  time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(g.start)),
                                  duration,
                                  request.headers.get('X-Forwarded-For', request.remote_addr),
                                  app_id,
                                  request.host.split(':', 1)[0],
                                  input_body,
                                  response.status_code,
                                  response.status,
                                  response.json)

        return response


def config_driver(config):
    if config.enable_es:
        elasticsearch_driver.initialize_driver(config.es_hosts, config.face_index)


def create_app(config):
    start_time = time.time()
    flask_app = Flask(__name__, static_folder='')
    LOGGER.info("Start API with config: {}".format(config.__dict__))

    flask_app.config.from_object(config)
    configure_logs(config)
    register_extensions(flask_app, config)
    register_blueprints(flask_app, config)
    configure_api(flask_app, config)
    config_driver(config)

    LOGGER.info("Start api time: {}s".format(time.time() - start_time))
    return flask_app
