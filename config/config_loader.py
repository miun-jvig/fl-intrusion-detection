from configparser import ConfigParser
import os

# load datasets
main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(main_folder, "config", "config.ini")
config_object = ConfigParser(comment_prefixes=('#', ';'), allow_no_value=True)
config_object.read(config_file)

# objects
server_cfg = config_object['SERVER']
client_cfg = config_object['CLIENT']
