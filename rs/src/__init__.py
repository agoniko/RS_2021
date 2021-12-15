import os
from pathlib import Path



ROOT_LOCATION = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

AUTH_LOCATION = os.path.join(os.path.join(ROOT_LOCATION, "resources"), "auth", "service-account.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = AUTH_LOCATION
