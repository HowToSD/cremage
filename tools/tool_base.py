"""
Base class for Tool.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional, Callable

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [PROJECT_ROOT, MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.gtk_utils import save_pil_image_by_file_chooser_dialog
from cremage.utils.gtk_utils import open_file_chooser_dialog

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768

class ToolBase(Gtk.Window):
    """
    Tool base.
    """       
    def __init__(
            self,
            title: Optional[str] = "Tool base",
            pil_image: Optional[Image.Image] = None,
            output_file_path: Optional[str] = None,
            save_call_back: Optional[Callable] = None,
            generation_information_call_back: Optional[Callable] = None,
            preferences: Optional[Dict[str, Any]] = None):
        """
        
        Args:
            title (str): Title of the window.
            pil_image (Image): The input PIL image.
            output_file_path (str): The name of the file to be saved in this method.
            save_call_back (callable): A function to call to send the image back to the caller.
                Function signature is the following:
                save_call_back(output_image:Image, generation_information:str)
                generation_information contains the serializedJSON dict
                with additional key "Additional process" with a value
                in List[string] format. Each method is to append the processing
                done in each method to this list.  E.g. if the image was cropped,
                it should add "cropped" to the list.
            generation_information_call_back (callable): A callback to obtain
                the generation information from the caller.
            preferences ([Dict[str, Any]]): The user's prefeerences.       
        """
        super().__init__(title=title)
        self.pil_image = pil_image
        self.output_pil_image = None
        self.save_call_back = save_call_back
        self.output_file_path = output_file_path
        self.generation_information_call_back = generation_information_call_back 
        self.preferences = dict() if preferences is None else preferences
        self.width = int(self.preferences["image_width"]) if "image_width" in self.preferences else DEFAULT_WIDTH
        self.height = int(self.preferences["image_height"]) if "image_height" in self.preferences else DEFAULT_HEIGHT
        self.set_default_size(self.width, self.height)

        # Get generation information from the caller.
        if self.generation_information_call_back:
            self.generation_information = self.generation_information_call_back()
        else:
            self.generation_information = None

        self.set_up_ui()   # This should be overridden in subclass

    def set_up_ui(self):
        """
        Sets up UI.

        This method should be overridden in subclass.
        """
        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create menu
        menubar = self.create_menu()
        vbox.pack_start(menubar, False, False, 0)

    def create_menu(self):
        """
        Creates the menu bar.
        """
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)
        
        menubar = Gtk.MenuBar()

        # File menu items
        filemenu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(filemenu)

        # File | Open
        open_item = Gtk.MenuItem(label="Open")  # Create open menu item
        open_item.connect("activate", self.on_open_clicked)  # Connect to handler
        open_item.add_accelerator("activate", accel_group, ord('O'),
                                Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
        filemenu.append(open_item)  # Add open item to file menu

        # File | Update the caller
        if self.save_call_back:
            update_item = Gtk.MenuItem(label="Update the caller")
            update_item.connect("activate", self.on_update_clicked)
            filemenu.append(update_item)

        # File | Save
        save_item = Gtk.MenuItem(label="Save")
        save_item.connect("activate", self.on_save_clicked)
        save_item.add_accelerator("activate", accel_group, ord('S'),
                                Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
        filemenu.append(save_item)
        
        # File | Exit
        exit_item = Gtk.MenuItem(label="Close")
        exit_item.connect("activate", self.close_window)
        filemenu.append(exit_item)

        menubar.append(file_item)
        return menubar
 
    def on_open_clicked(self, widget):
        """
        Handles the load mask image button press event.
        """
        image_path = open_file_chooser_dialog(self, title="Select an image file")
        if image_path:
            self.pil_image = Image.open(image_path)

    def on_update_clicked(self, widget):
        """
        Saves output_pil_image in the file specified by the caller (self.output_path).
        """
        if self.output_pil_image and self.output_file_path:
            self.output_pil_image.save(self.output_file_path)
            if self.save_call_back is not None:
                self.save_call_back(self.output_pil_image, self.generation_information)

    def on_save_clicked(self, widget):
        """
        Saves output_pil_image.
        """
        if self.output_pil_image:
            file_path = save_pil_image_by_file_chooser_dialog(self, self.output_pil_image)
            if file_path:
                if self.save_call_back is not None:
                    self.save_call_back(self.output_pil_image, self.generation_information)
           
    def close_window(self, widget):
        self.close()


def main():
    app = ToolBase()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()