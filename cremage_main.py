import os
import logging
import platform

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

os_name = platform.system()
if os_name == 'Darwin':  # macOS
    gpu_device = "mps"
else:
    gpu_device = "cuda"

os.environ["GPU_DEVICE"] = gpu_device

import set_up_path
from cremage.configs.preferences import load_user_config
user_config = load_user_config()
if user_config["enable_hf_internet_connection"] == True:
    os.environ["ENABLE_HF_INTERNET_CONNECTION"] = "1"
else:
    os.environ["ENABLE_HF_INTERNET_CONNECTION"] = "0"
from cremage.ui.initializer import initializer


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class MainWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Cremage")
        initializer(self)


def main():
    win = MainWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
