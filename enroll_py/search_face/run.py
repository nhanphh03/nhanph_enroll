import sys
sys.path.insert(0, "./modules/inference")

from config_run import Config, init_config
from app import create_app
from os import environ

config_path = environ.get('CUSTOM_CONFIG_PATH', 'config/production.yml')

get_config_mode = environ.get("CONFIG_MODE", "Development")
try:
    if get_config_mode.capitalize() == 'Development':
        config_path = 'config/development.yml'
    init_config(config_path)
    flask_app = create_app(Config)

except KeyError as e:
    print(e)
    exit("Error: Invalid CONFIG_MODE environment variable entry. (Production/Development)")

if __name__ == '__main__':
    flask_app.run("0.0.0.0", port=15000, debug=Config.DEBUG)
