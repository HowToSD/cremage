"""
Utility functions related to handling images
"""
import os
import sys
import io
from typing import List, Any, Dict

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, Gdk
import cairo

from PIL import Image

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.image_utils import resize_with_padding, pil_image_to_gtk_image, pil_image_to_pixbuf
from cremage.utils.image_utils import resize_pil_image

def create_center_aligned_button(label=""):
    """
    Creates a center-aligned button.

    Args:
        label(str): The button label.
    Returns:
        Tuple of the button object and button wrapper object.
        Use the button wrapper object to add to the container.
    """
    button_wrapper = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    hbox1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    hbox2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    button = Gtk.Button(label=label)

    button_wrapper.pack_start(hbox1,  
                    True,  # Expand
                    True,  # Takes up the space
                    0)  # No padding
    button_wrapper.pack_start(button,
                    False,  # Do not expand
                    False,  # Do not takes up the space
                    0)  # No padding
    button_wrapper.pack_start(hbox2,
                    True,  # Expand
                    True,  # Takes up the space
                    0)  # No padding
    return button, button_wrapper


def create_surface_from_pil(pil_image: Image):
    # Convert PIL image to GdkPixbuf
    width, height = pil_image.size
    mode = pil_image.mode

    has_alpha = mode == "RGBA"

    data = pil_image.tobytes()
    
    pixbuf = GdkPixbuf.Pixbuf.new_from_data(
        data,
        GdkPixbuf.Colorspace.RGB,
        has_alpha,
        8,  # bits per sample
        width,
        height,
        width * (4 if has_alpha else 3)  # row stride
    )
    
    # Create Cairo surface from GdkPixbuf
    image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32 if has_alpha else cairo.FORMAT_RGB24, width, height)
    cairo_context = cairo.Context(image_surface)
    Gdk.cairo_set_source_pixbuf(cairo_context, pixbuf, 0, 0)
    cairo_context.paint()
    
    return image_surface


def show_info_dialog(win: Gtk.Window, message: str) -> None:
    """
    Displays an information message.

    Args:
        win (Gtk.Window): The parent window.
        message (str): The message to display.
    """
    dialog = Gtk.MessageDialog(win, 0, Gtk.MessageType.INFO,
                                Gtk.ButtonsType.CLOSE, message)
    dialog.run()
    dialog.destroy()


def show_error_dialog(win: Gtk.Window, message: str) -> None:
    """
    Displays an error message.

    Args:
        win (Gtk.Window): The parent window.
        message (str): The message to display.
    """
    dialog = Gtk.MessageDialog(win, 0, Gtk.MessageType.ERROR,
                                Gtk.ButtonsType.CLOSE, message)
    dialog.run()
    dialog.destroy()


def set_pil_image_to_gtk_image(pil_image:Image, gtk_image: Gtk.Image):
    pixbuf = pil_image_to_pixbuf(pil_image)
    gtk_image.set_from_pixbuf(pixbuf)


def set_image_to_gtk_image_from_file(*,
                                file_path:str= None,
                                gtk_image:Gtk.Image = None,
                                target_edge_length:int=256) -> Gtk.Image:
    pil_image = Image.open(file_path)
    resized_pil_image = resize_pil_image(pil_image, target_edge_length)
    set_pil_image_to_gtk_image(resized_pil_image, gtk_image)


def open_file_chooser_dialog(parent: Gtk.Window,
                             title:str="Choose a file to open",
                             file_type_dict:Dict[str, str] = {
                                 "PNG files":"image/png",
                                 "JPG files":"image/jpeg"
                             },
                             ) -> str:
    """
    Opens a file chooser dialog with a specified parent window and a customizable title and file filters. 
    This dialog provides a GUI for selecting a file to open with specific filters for file types.

    Args:
        parent (Gtk.Window): The Gtk.Window that acts as the parent of the file chooser dialog. This window is typically the application's main window.
        title (str, optional): The title of the file chooser dialog. Defaults to "Choose a file to open".
        file_type_dict (dict of str: str, optional): A dictionary where the keys represent the name of the filter, as shown in the dialog, and the values are the corresponding MIME types. Defaults to {"PNG files": "image/png", "JPG files": "image/jpeg"}.

    Returns:
        The file name selected (str). If the file was not selected, None will be returned.
    """
    dialog = Gtk.FileChooserDialog(title=title, parent=parent,
                                    action=Gtk.FileChooserAction.OPEN)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

    for name, mime_type in file_type_dict.items():
        file_filter = Gtk.FileFilter()
        file_filter.set_name(name)
        file_filter.add_mime_type(mime_type)
        dialog.add_filter(file_filter)
    response = dialog.run()
    file_name = None
    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
    dialog.destroy()
    return file_name


def save_pil_image_by_file_chooser_dialog(parent: Gtk.Window, pil_image: Image) -> bool:
    """
    Displays a file chooser dialog to save a PIL image.

    Args:
        parent (Gtk.Window): The Gtk.Window that acts as the parent of the file chooser dialog. This window is typically the application's main window.
        pil_image (Image): The PIL format image to save.

    Returns:
        The name of the file that was saved. None if the file was not saved.
    """
    # Show file chooser dialog
    chooser = Gtk.FileChooserDialog(title="Save File", 
                                    parent=parent, 
                                    action=Gtk.FileChooserAction.SAVE)
    # Add cancel button            
    chooser.add_buttons(Gtk.STOCK_CANCEL, 
                        Gtk.ResponseType.CANCEL, 
                        Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    response = chooser.run()
    if response == Gtk.ResponseType.OK:
        filename = chooser.get_filename()
        pil_image.save(filename)
        retval = filename
    else:
        retval = None
    chooser.destroy()

    return retval


def get_full_path_from_combo_box(dir_path: str, combobox: Gtk.ComboBoxText):
    """
    Returns the full path from the input directory and the combobox.

    Args:
        dir_path (str): The full path of the directory.
        combobox (Gtk.ComboBoxText): The combo box containing the list of files and "None". 
    """
    cb_text = combobox.get_active_text()
    return "None" if cb_text == "None" or cb_text == "" else os.path.join(dir_path, cb_text)  


def resized_gtk_image_from_pil_image(
                                pil_image:Image.Image,
                                target_width:int=None,
                                target_height:int=None) -> Gtk.Image:
    return pil_image_to_gtk_image(
        resize_with_padding(pil_image, target_width=target_width, target_height=target_height))

def resized_gtk_image_from_file(*,
                                file_path:str= None,
                                target_width:int=None,
                                target_height:int=None) -> Gtk.Image:
    pil_image = Image.open(file_path)
    return pil_image_to_gtk_image(
        resize_with_padding(pil_image, target_width=target_width, target_height=target_height))


def text_view_get_text(tv: Gtk.TextView) -> str:
    textbuffer = tv.get_buffer()
    start_iter = textbuffer.get_start_iter()
    end_iter = textbuffer.get_end_iter()
    text = textbuffer.get_text(start_iter, end_iter, True)  # True for including hidden characters
    return text


def text_view_set_text(tv: Gtk.TextView, text: str) -> None:
    buffer = tv.get_buffer()
    if text is None:
        text = ""
    buffer.set_text(text)


# Helper function to update a dropdown list
def update_combo_box(combo, options, active_index=0):
    """
    Update the ComboBox with new options.

    Parameters:
    - combo: The Gtk.ComboBoxText instance to update.
    - options: A list of new options to add to the ComboBox.
    - active_index: The index of the item to set as active.
    """
    if isinstance(combo, Gtk.ComboBox):  # special typeahead combobox for Cremage
        # Get the ListStore from the ComboBox
        model = combo.get_model()
        model.clear()  # Clear all existing items in the ListStore
        # Add new items to the ListStore
        for option in options:
            model.append([str(option)])
        # Set the active item
        combo.set_active(active_index)
    else:
        combo.remove_all()  # Remove all existing items
        for option in options:
            combo.append_text(str(option))  # Add new items
        combo.set_active(active_index)  # Set the active item

def create_combo_box(options: List[Any], active_index=0):
    combo = Gtk.ComboBoxText()
    for option in options:
        combo.append_text(str(option))
    combo.set_active(active_index)
    return combo

def create_combo_box_typeahead(options: List[Any], active_index=0, width=None):

    # Create a ListStore with one string column to use as the model
    liststore = Gtk.ListStore(str)
    for option in options:
        liststore.append([option])

    # Create the ComboBox, but with an entry
    combobox = Gtk.ComboBox.new_with_entry()
    combobox.set_model(liststore)
    combobox.set_entry_text_column(0)

    if width:
        combobox.set_size_request(width, -1)

    # Setup the completion for the entry
    entry = combobox.get_child()
    completion = Gtk.EntryCompletion()
    completion.set_model(liststore)
    completion.set_text_column(0)
    entry.set_completion(completion)

    # Filter function to match text typed by the user
    def match_func(completion, key_string, iter, data):
        model = completion.get_model()
        model_string = model[iter][0].lower()
        if key_string.lower() in model_string:
            return True
        return False

    completion.set_match_func(match_func, None)

    initial_text = options[active_index]
    combobox.get_child().set_text(str(initial_text))
    return combobox



class AlertDialog(Gtk.Dialog):
    def __init__(self, parent, message):
        super().__init__(title="Alert", transient_for=parent, flags=0)
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)
        self.set_default_size(150, 100)
        self.set_modal(True)  # Make the dialog modal

        # Create a label to display the message
        label = Gtk.Label(label=message)

        # Add the label to the content area of the dialog
        box = self.get_content_area()
        box.add(label)
        self.show_all()


def show_alert_dialog(message):
    # Create a new GTK window
    win = Gtk.Window()
    win.connect("destroy", Gtk.main_quit)
    
    # Create and show the alert dialog
    dialog = AlertDialog(win, message)
    response = dialog.run()
    
    # Handle the response if needed, then destroy the dialog
    print(f"Response: {response}")
    dialog.destroy()
