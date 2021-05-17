from os.path import join

DEFAULT_GUI = False
DEFAULT_NET_FILE = join("configs", "two_inter", "two_inter.net.xml")
DEFAULT_RAND_ROUTES_ON_RESET = True
DEFAULT_RANKED = True


def get_env_config(**kwargs):
    config = {
        "gui": kwargs.get("gui", DEFAULT_GUI),
        "net-file": kwargs.get("net-file", DEFAULT_NET_FILE),
        "rand_routes_on_reset": kwargs.get("rand_routes_on_reset", DEFAULT_RAND_ROUTES_ON_RESET),
        "ranked": kwargs.get("ranked", DEFAULT_RANKED),
    }
    return config