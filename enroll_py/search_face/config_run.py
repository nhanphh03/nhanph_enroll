import os

import yaml


class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))
    log_config_file = os.path.join(basedir, "logger.yaml")
    log_folder = os.path.join(basedir, "logs")
    log_images = os.path.join(basedir, "logs", "images")

    # ES database config
    enable_es = True
    api_log_index = "api_log"
    face_index = 'face'
    vector_index = 'vector'
    es_hosts = ["localhost:9200"]

    # Face config
    similar_thresh_search = 0.55
    max_output_search = 10
    similar_thresh_compare = 0.7
    similar_thresh_same = 0.95
    similar_thresh_liveness = 0.6
    thresh_liveness = 0.6
    min_liveness = 0

    # face anti spoof
    face_anti_spoof_image_size = 224
    face_anti_spoof_threshold = 0.8

    # Check action config
    thresh_action = 1

    thresh_up = -10
    thresh_down = 10
    thresh_left = 10
    thresh_right = -10
    thresh_tilting_left = 8
    thresh_tilting_right = -8
    thresh_blink = 5.0
    thresh_mouth = 0.7

    blueprints = []
    apis = ['face', 'face2']

    DEBUG = False
    monitor = False

    enable_log = False

    service_host = "localhost"
    service_port = 8500
    service_timeout = 10

    @staticmethod
    def init_log(log_config_file, log_folder, log_images):
        Config.log_config_file = os.path.join(Config.basedir, log_config_file)
        Config.log_folder = os.path.join(Config.basedir, log_folder)
        Config.log_images = os.path.join(Config.basedir, Config.log_folder, log_images)


def init_config(config_path):
    print(f"Load config from: {config_path}")
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f.read())
        Config.init_log(cfg_dict['log_config_file'],
                        cfg_dict['log_folder'],
                        cfg_dict['log_images'])

        Config.enable_es = cfg_dict['enable_es']
        Config.api_log_index = cfg_dict['api_log_index']
        Config.face_index = cfg_dict['face_index']
        Config.es_hosts = cfg_dict['es_hosts']

        Config.similar_thresh_compare = cfg_dict['similar_thresh_compare']
        Config.similar_thresh_liveness = cfg_dict['similar_thresh_liveness']
        Config.thresh_liveness = cfg_dict['thresh_liveness']
        if 'min_liveness' in cfg_dict:
            Config.min_liveness = cfg_dict['min_liveness']

        if 'thresh_action' in cfg_dict:
            Config.thresh_action = cfg_dict['thresh_action']

        if 'thresh_up' in cfg_dict:
            Config.thresh_up = cfg_dict['thresh_up']
        if 'thresh_down' in cfg_dict:
            Config.thresh_down = cfg_dict['thresh_down']
        if 'thresh_left' in cfg_dict:
            Config.thresh_left = cfg_dict['thresh_left']
        if 'thresh_right' in cfg_dict:
            Config.thresh_right = cfg_dict['thresh_right']
        if 'thresh_tilting_left' in cfg_dict:
            Config.thresh_tilting_left = cfg_dict['thresh_tilting_left']
        if 'thresh_tilting_right' in cfg_dict:
            Config.thresh_tilting_right = cfg_dict['thresh_tilting_right']
        if 'thresh_blink' in cfg_dict:
            Config.thresh_blink = cfg_dict['thresh_blink']
        if 'thresh_mouth' in cfg_dict:
            Config.thresh_mouth = cfg_dict['thresh_mouth']

        if 'face_anti_spoof_image_size' in cfg_dict:
            Config.face_anti_spoof_image_size = cfg_dict['face_anti_spoof_image_size']

        if 'face_anti_spoof_threshold' in cfg_dict:
            Config.face_anti_spoof_threshold = cfg_dict['face_anti_spoof_threshold']

        Config.blueprints = cfg_dict['blueprints']
        Config.apis = cfg_dict['apis']

        Config.service_host = cfg_dict['service_host']
        Config.service_port = cfg_dict['service_port']
        Config.service_timeout = cfg_dict['service_timeout']

        Config.DEBUG = cfg_dict['DEBUG']

        print(f"============> Loading config: \n{cfg_dict}")
