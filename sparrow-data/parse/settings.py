import sys
import os

# Add the path to the betterbrain settings to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(f"Parent dir: {parent_dir}")
betterbrain_settings_path = os.path.join(parent_dir)
if betterbrain_settings_path not in sys.path:
    print(f"Adding {betterbrain_settings_path} to sys.path")
    sys.path.insert(0, betterbrain_settings_path)



betterbrain_settings_path = os.path.join(parent_dir, 'evaluation')
if betterbrain_settings_path not in sys.path:
    print(f"Adding {betterbrain_settings_path} to sys.path")
    sys.path.insert(0, betterbrain_settings_path)


import betterbrain.betterbrain.settings as sparrow_settings
from eval_utils import init_django
import scripts.aws.spot.aws_entrypoint as aws_entrypoint



# print(betterbrain_settings_path)
# import importlib.util
# import sys
# import os

# # Dynamically import the betterbrain.settings module from its file path
# settings_file = os.path.join(betterbrain_settings_path, "settings.py")
# spec = importlib.util.spec_from_file_location("sparrow_settings", settings_file)
# sparrow_settings = importlib.util.module_from_spec(spec)
# sys.modules["sparrow_settings"] = sparrow_settings
# spec.loader.exec_module(sparrow_settings)