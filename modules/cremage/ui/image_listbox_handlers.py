"""
Defines handlers for the image listbox.

# Description of key variables:
app.current_image_start_index
    An index to indicate the image that is shown at the top slot of the list box.
    This can take any value from zero to the image count in the output directory - 1.
app.total_images:
    Total number of images in the output directory
app.images_per_page
    Number of images to be shown at the same time on screen.
"""
import os
import logging
import sys
import shutil
import json
import time
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from cremage.const.const import MAIN_IMAGE_CANVAS_SIZE
from cremage.utils.image_utils import pil_image_to_gtk_image
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.image_utils import load_resized_pil_image
from cremage.utils.image_utils import get_png_paths
from cremage.utils.gtk_utils import text_view_set_text

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def update_image_file_list_to_show_generated_images(app):
    refresh_image_file_list(app)
    app.current_image_start_index = 0
    update_listbox_all(app, app.current_image_start_index)

def refresh_image_file_list(app, skip_adjustment=False):
    if app.output_dir is None:
        app.total_images = 0
        app.total_positions = 0
        return

    app.image_paths = get_png_paths(app.output_dir)
    app.total_images = len(app.image_paths)
    # Adjust the total_positions calculation to account for individual steps
    app.total_positions = app.total_images - app.images_per_page

    if skip_adjustment is False:
        app.adjustment.set_upper(app.total_positions)


def image_listbox_key_press_handler(app:Gtk.Window, widget:Gtk.ListBox, event:Gdk.EventKey):
    """
    Handles key press events for a specific Gtk.ListBox widget in the application.

    This function processes key press events to implement functionality such as navigation,
    selection, and other custom actions within an image listbox.

    Args:
        app (Gtk.Window): The Window instance, which holds the state and control logic of the app.
        widget (Gtk.ListBox): The list box widget that received the event. This is where the key press event is handled.
        event (Gdk.EventKey): The key press event object, which contains data about the specific key press, like the key code and modifiers.

    Returns:
        bool: True if the event was handled and should not propagate further; False otherwise.
    """
    keyname = Gdk.keyval_name(event.keyval)
    cmd = (event.state & Gdk.ModifierType.META_MASK) != 0  # Command key
    fn = (event.state & Gdk.ModifierType.MOD2_MASK) != 0  # Often Fn key
    ctrl = (event.state & Gdk.ModifierType.CONTROL_MASK) != 0
    shift = (event.state & Gdk.ModifierType.SHIFT_MASK) != 0

    if keyname in ['m', 'M']:
        print("M was pressed")
        selected_row = app.listbox.get_selected_row() # obj 0-7
        mark_image(app, selected_row.get_index())
        return True  # Return True if you want to stop other handlers from being invoked for this event.

    if keyname in ['g', 'G']:
        print("G was pressed")
        go_to_marked_image(app)
        return True

    if keyname in ['f', 'F']:
        print("F was pressed")
        selected_row = app.listbox.get_selected_row() # obj 0-7
        copy_image_to_favorites(app, selected_row.get_index())
        return True

    if keyname == "Delete":
        selected_row = app.listbox.get_selected_row() # obj 0-7
        delete_image(app, selected_row.get_index())
        return True
    
    elif keyname == "BackSpace":
        selected_row = app.listbox.get_selected_row() # obj 0-7
        delete_image(app, selected_row.get_index())
        return True
    
    elif keyname == "Home":
        return handle_home_key(app)

    elif keyname == "End":
        return handle_end_key(app)

    elif keyname == "Up":
        if cmd or fn:  # For Mac
            return handle_home_key(app)

        # Note that this is called before the selection moves
        # to the new row, so the selected row is still the current row
        # before update
        selected_row = app.listbox.get_selected_row() # obj 0-7
        logger.debug(f"List index before key was pressed: {selected_row.get_index()}")

        if selected_row is None:  # If row is not selected, do not process further
            return True

        if app.current_image_start_index > 0:  # global image index
            if selected_row.get_index() == 0:  # first on the list, so we need to scroll
                app.current_image_start_index -= 1

                # Items to be on the listglobal index start
                update_listbox_up(app, app.current_image_start_index)
                new_row = app.listbox.get_row_at_index(0)    # Get the first item on the list
                app.listbox.select_row(new_row)              # and select it
                new_row.grab_focus()                         # and move focus there
                update_main_image(app, new_row.get_index())  # Display the image on main canvas
                return True  # Prevent the default beep sound
            else:
                update_main_image(app, selected_row.get_index()-1)
                return False  # Let Gtk move the selection
        else:  # Currently, the very first image is on the first slot of the list
            if selected_row.get_index() > 0:
                update_main_image(app, selected_row.get_index()-1)
                return False  # Let Gtk move the selection
            else:  # First image is already selected, so no op
                return True
    elif keyname == "Down":
        if cmd or fn:  # For Mac
            return handle_end_key(app)

        # Note that this is called before the selection moves
        # to the new row, so the selected row is still the current row
        # before update
        selected_row = app.listbox.get_selected_row()
        if selected_row is None:  # If row is not selected, do not process further
            return True
        logger.debug(selected_row.get_index())

        # If number of total images is 20, and the list size is 8,
        # index 12 is the largest index.
        # If index 12 is on the first slot, index 12-19 will be displayed.
        # So if the current_image_start_index is smaller than 12,
        # it means that the list box can be scrollable (or content is updatable).

        # If the list box is updatable,
        if app.current_image_start_index < app.total_images - app.images_per_page:

            # If the selected row before down key is pressed is the last item on the list
            if selected_row.get_index() == app.images_per_page - 1:
                logger.debug("down key pressed on last row")
                app.current_image_start_index += 1
                update_listbox_down(app, app.current_image_start_index)

                # Select and set focus on the newly added row
                new_row = app.listbox.get_row_at_index(app.images_per_page - 1) # row 7
                app.listbox.select_row(new_row)
                new_row.grab_focus()
                update_main_image(app, new_row.get_index())
                return True  # Prevent the default beep sound
            else:
                update_main_image(app, selected_row.get_index() + 1)
                return False  # Let Gtk move the selection
        else:  # last page
            if selected_row.get_index() <= app.images_per_page - 2:
                update_main_image(app, selected_row.get_index() + 1)
                return False  # Let Gtk move the selection
            else:
                return True

    return False


def on_row_activated(app:Gtk.Window, listbox: Gtk.ListBox, listboxrow: Gtk.ListBoxRow) -> None:
    """
    Event handler for the left mouse button click on a list item.

    Args:
        listbox (Gtk.ListBox): The ListBox containing the row.
        listboxrow (Gtk.ListBoxRow): The row that was activated.
    """
    logger.debug(f"Row {listboxrow.get_index()} activated")
    update_main_image(app, listboxrow.get_index())


def update_main_image(app:Gtk.Window, listbox_index:int) -> None:
    """
    Updates the main image when a different image is selected in the image list
    view on the UI.

    Note: current_image_start_index is the index of the image that is
    currently display on the top row of the listbox
    """
    target_size = MAIN_IMAGE_CANVAS_SIZE  # longer edge length
    current_image_index = app.current_image_start_index + listbox_index
    image_path = app.image_paths[current_image_index]

    # Original size
    tmp_image = Image.open(image_path)
    app.current_image = tmp_image.copy()
    tmp_image.close()

    # Resized image
    pil_image = load_resized_pil_image(image_path, target_size=target_size)

    pixbuf = pil_image_to_pixbuf(pil_image)
    app.image.set_from_pixbuf(pixbuf)
    generation_parameters = pil_image.info["generation_data"] \
        if "generation_data" in pil_image.info is not None and \
            "generation_data" in pil_image.info and \
            pil_image.info["generation_data"] is not None \
        else ""
    text_view_set_text(app.generation_information, generation_parameters)
    if generation_parameters:
        app.current_image_generation_information_dict = json.loads(generation_parameters)
    else:
        app.current_image_generation_information_dict = None


def mark_image(app:Gtk.Window, listbox_index:int):
    """
    Marks current image so that the user can go back.
    This is helpful when the user has a large number of images and mark an image
    to go back for a later viewing.
    
    Args:
        list_box_index (int):  Position in the listbox of the image to be deleted.
            0-based.
    """
    # Get the global image index of the image to be deleted
    current_image_index = app.current_image_start_index + listbox_index
    image_path = app.image_paths[current_image_index]
    app.marked_image_path = image_path


def go_to_marked_image(app:Gtk.Window) -> bool :
    """
    Moves the selection to the marked image in the image list box.

    Args:
        app (Gtk.Window): Main application.
    Returns:
        True to indicate that the key press event will not propate further.
    """
    # Find the global image index
    if app.marked_image_path is None or os.path.exists(app.marked_image_path) is False:
        return True  # No op
    
    # Find the global index
    # TODO: Optimize
    for i, path in enumerate(app.image_paths):
        if path == app.marked_image_path:
            app.current_image_start_index = i

    update_listbox_all(app, app.current_image_start_index)
    first_row = app.listbox.get_children()[0]
    app.listbox.select_row(first_row)
    first_row.grab_focus()
    update_main_image(app, first_row.get_index())
    return True

def copy_image_to_favorites(app:Gtk.Window, listbox_index:int):
    """
    Copies image to the favorites folder.

    Args:
        list_box_index (int):  Position in the listbox of the image to be deleted.
            0-based.
    """
    # Get the global image index of the image to be deleted
    current_image_index = app.current_image_start_index + listbox_index
    image_path = app.image_paths[current_image_index]
    base_name = os.path.basename(image_path)
    target_file_path = os.path.join(app.favorites_dir, base_name)
    if os.path.exists(target_file_path):
        # If target path already exists in a trash directory,
        # create a new file name by appending the time to the base name of the file.
        target_file_path = os.path.splitext(target_file_path)[0] + str(time.time()) + os.path.splitext(target_file_path)[1]

    # Move the file to trash dir
    shutil.copy(image_path, target_file_path)


def delete_image(app:Gtk.Window, listbox_index:int):
    """
    Note: current_image_start_index is the index of the image that is
    currently display on the top row of the listbox

    Args:
        list_box_index (int):  Position in the listbox of the image to be deleted.
            0-based.
    """
    # Get the global image index of the image to be deleted
    current_image_index = app.current_image_start_index + listbox_index
    image_path = app.image_paths[current_image_index]
    base_name = os.path.basename(image_path)
    target_file_path = os.path.join(app.trash_dir, base_name)
    if os.path.exists(target_file_path):
        # If target path already exists in a trash directory,
        # create a new file name by appending the time to the base name of the file.
        target_file_path = os.path.splitext(target_file_path)[0] + str(time.time()) + os.path.splitext(target_file_path)[1]

    # Move the file to trash dir
    shutil.move(image_path, target_file_path)

    # Update the list of image files
    refresh_image_file_list(app)
    if app.total_images == 0:
        return

    if app.current_image_start_index >= app.total_images:
        app.current_image_start_index -= 1

    update_listbox_all(app, app.current_image_start_index)

    # Check to see if the listbox contains the same number
    list_item_count = len(app.listbox.get_children())
    if list_item_count == 0:
        # TODO: Set black image
        return
    elif listbox_index + 1 <= list_item_count:
        new_index = listbox_index
    else:
        new_index = listbox_index - 1
    new_row = app.listbox.get_row_at_index(new_index)
    app.listbox.select_row(new_row)
    new_row.grab_focus()
    update_main_image(app, new_row.get_index())


def on_scrollbar_value_changed(app:Gtk.Window, adjustment:int):
    start_index = int(adjustment.get_value())
    update_listbox_all(app, start_index)


def get_gtk_image_for_index(app:Gtk.Window, image_index:int):
    """
    image_index: Global image index in the directory
    """
    target_size = 128  # longer edge length
    image_path = app.image_paths[image_index]
    pil_image = load_resized_pil_image(image_path)
    return pil_image_to_gtk_image(pil_image)


def update_listbox_all(app:Gtk.Window, image_start_index:int) -> None:
    """
    Populates the image list box with images corresponding to the image_start_index.

    Args:
        app (Gtk.Window): Main application window.
        image_start_index (int): Global image index in the image output directory.
    """
    app.current_image_start_index = image_start_index

    # Remove all list items
    for child in app.listbox.get_children():
        app.listbox.remove(child)

    # Populate the list box images.
    # Normally, if the list contain 10 images, we should put 10 images.
    # However, if you are close to the bottom of the list, then there may not be 10 images.
    # Therefore, check the total number of images, and pick the smaller one
    # as the end image index.
    # Case 1: You are not close to the bottom of the list
    #     start_index = 3000 (3001st to 3010th or indices: 3000 to 3009)
    #     per_page = 10
    #     end_index_plus_1 = 3010
    # Case 2: You are close to the bottom
    #     start index = 4995 (4996th to 5000th or indices 4995 to 4999)
    #     per_page = 10
    #     end_index_plus_1 = 5000
    end_index_plus_1 = min(image_start_index + app.images_per_page, app.total_images)
    for i in range(image_start_index, end_index_plus_1):
        logger.debug(f"image_start_index: {image_start_index}")
        logger.debug(f"end_index_plus_1: {end_index_plus_1}")
        logger.debug(f"i: {i}")

        row = Gtk.ListBoxRow()
        gtk_image = get_gtk_image_for_index(app, i)
        row.add(gtk_image)
        app.listbox.add(row)

    app.listbox.show_all()


def update_listbox_down(app:Gtk.Window, start_index:int):
    items = app.listbox.get_children()
    if len(items) >= app.images_per_page:
        app.listbox.remove(items[0])
    logger.debug(f"start_index: {start_index}")

    # end_index is 0-based
    end_index = min(start_index + app.images_per_page - 1, app.total_images - 1)
    logger.debug(f"end_index: {end_index}")
    row = Gtk.ListBoxRow()

    gtk_image = get_gtk_image_for_index(app, end_index)
    row.add(gtk_image)
    app.listbox.add(row)
    app.listbox.show_all()


def update_listbox_up(app:Gtk.Window, start_index:int):
    """
    Args:
        start_index: Global image index [0, 5000)
    """
    items = app.listbox.get_children()
    if len(items) >= app.images_per_page:
        # remove last item
        app.listbox.remove(items[-1])

    start_index = max(start_index, 0)
    row = Gtk.ListBoxRow()

    gtk_image = get_gtk_image_for_index(app, start_index)
    row.add(gtk_image)

    app.listbox.insert(row, 0)
    app.listbox.show_all()


def handle_home_key(app:Gtk.Window) -> bool :
    """
    Handles the Home key press.
    Moves the selection to the first item in the image list box.

    Args:
        app (Gtk.Window): Main application.
    Returns:
        True to indicate that the key press event will not propate further.
    """
    app.current_image_start_index = 0
    update_listbox_all(app, app.current_image_start_index)
    first_row = app.listbox.get_children()[0]
    app.listbox.select_row(first_row)
    first_row.grab_focus()
    update_main_image(app, first_row.get_index())
    return True


def handle_end_key(app:Gtk.Window) -> bool:
    """
    Handles the End key press.
    Moves the selection to the last item in the image list box.

    Note that in order to return True, it does the new row selection as well as setting
    focus to the new row which are normally handled by Gtk.

    Args:
        app (Gtk.Window): Main application.

    Returns:
        True to indicate that the key press event will not propate further.
    """
    if app.total_images < app.images_per_page:
        pass

    # e.g. 5000 images and image listbox contains 10,
    # start index should be 4990 to show 4991st to 5000th images
    app.current_image_start_index = app.total_images - app.images_per_page

    update_listbox_all(app, app.current_image_start_index)
    last_row = app.listbox.get_children()[len(app.listbox.get_children())-1]
    app.listbox.select_row(last_row)
    last_row.grab_focus()
    update_main_image(app, last_row.get_index())
    return True  # Do not propagate further