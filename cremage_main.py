import os
import sys
import multiprocessing
try:
    multiprocessing.set_start_method('spawn') # This has to be called here. Do not move this line.
except:
    print("Ignoring set_start_method error.")
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

sys.path = ["modules"] + sys.path
from cremage.configs.preferences import load_user_config
user_config = load_user_config()
if user_config["enable_hf_internet_connection"] == True:
    os.environ["ENABLE_HF_INTERNET_CONNECTION"] = "1"
else:
    os.environ["ENABLE_HF_INTERNET_CONNECTION"] = "0"
from cremage.ui.initializer import initializer
from cremage.ui.initializer import setup_field_visibility
from cremage.mp.mp import init_mp, MP_MESSAGE_TYPE_EXIT

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class MainWindow(Gtk.Window):
    def __init__(self, ui_to_ml_queue, ml_to_ui_queue):
        super().__init__(title="Cremage")
        initializer(self, ui_to_ml_queue, ml_to_ui_queue)

def process_exit(widget, event=None):
    ui_to_ml_queue.put({"type": MP_MESSAGE_TYPE_EXIT})
    Gtk.main_quit()

def main():
    global ui_to_ml_queue
    ui_to_ml_queue, ml_to_ui_queue = init_mp()
    win = MainWindow(ui_to_ml_queue, ml_to_ui_queue)
    win.connect("destroy", process_exit)
    win.show_all()
    setup_field_visibility(win)
    Gtk.main()


if __name__ == "__main__":
    main()
