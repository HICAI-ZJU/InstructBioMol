import yaml
import logging

logger = logging.getLogger(__name__)



def load_config(args):
    '''the configuration of each model can rewrite the base configuration'''
    # base config
    base_configuration = load_base_config()

    configuration = base_configuration
    return configuration


def load_base_config():
    config_path = f'config/base.yaml'
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f'[!] load base configuration: {config_path}')
    return configuration
