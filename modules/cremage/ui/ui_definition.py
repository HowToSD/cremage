"""
Main UI definition
"""
import os
import logging
import sys
import time
import re
from collections import OrderedDict
from functools import partial

import omegaconf
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from cremage.const.const import MAIN_IMAGE_CANVAS_SIZE, TRUE_FALSE_LIST, FACE_MODEL_NAME
from cremage.const.const import THUMBNAIL_IMAGE_EDGE_LENGTH
from cremage.const.const import GENERATOR_MODEL_TYPE_LIST, FACE_DETECTION_METHOD_LIST
from cremage.ui.generate_handler import generate_handler
from cremage.ui.save_preference_handler import save_preference_handler
from cremage.ui.graffiti_editor_widget_click_handler import graffiti_editor_widget_click_handler
from cremage.ui.control_image_view_click_handler import control_image_view_click_handler
from cremage.ui.generation_mode_toggle_handler import generation_mode_toggle_handler
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.gtk_utils import text_view_get_text, show_error_dialog
from cremage.utils.gtk_utils import text_view_set_text
from cremage.utils.gtk_utils import create_combo_box, create_combo_box_typeahead
from cremage.utils.gtk_utils import resized_gtk_image_from_file
from cremage.ui.image_listbox_handlers import image_listbox_key_press_handler
from cremage.ui.image_listbox_handlers import on_row_activated
from cremage.ui.image_listbox_handlers import on_scrollbar_value_changed
from cremage.ui.image_listbox_handlers import update_listbox_all
from cremage.ui.model_path_update_handler import update_ldm_model_name_value_from_ldm_model_dir
from cremage.ui.model_path_update_handler import update_ldm_inpaint_model_name_value_from_ldm_model_dir
from cremage.ui.model_path_update_handler import update_vae_model_name_value_from_vae_model_dir
from cremage.ui.model_path_update_handler import update_control_model_name_value_from_control_model_dir
from cremage.ui.model_path_update_handler import update_lora_model_name_value_from_lora_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_ldm_inpaint_model_name_value_from_sdxl_ldm_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_vae_model_name_value_from_sdxl_vae_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_lora_model_name_value_from_sdxl_lora_model_dir
from cremage.ui.model_path_update_handler import update_pixart_sigma_ldm_model_name_value_from_pixart_sigma_ldm_model_dir
from cremage.ui.input_image_view_click_handler import input_image_view_click_handler
from cremage.ui.main_image_view_click_handler import main_image_view_click_handler
from cremage.ui.face_input_image_view_click_handler import face_input_image_view_click_handler
from cremage.ui.face_input_image_view_click_handler import face_input_image_view_close_handler
from cremage.ui.mask_image_view_click_handler import mask_image_view_click_handler
from cremage.ui.copy_image_widget_click_handler import copy_image_widget_click_handler
from cremage.ui.drag_and_drop_handlers import control_image_drag_data_received
from cremage.ui.drag_and_drop_handlers import input_image_drag_data_received
from cremage.ui.drag_and_drop_handlers import main_image_drag_data_received
from cremage.ui.drag_and_drop_handlers import face_input_image_drag_data_received
from cremage.ui.menu_definition import main_menu_definition
from cremage.ui.image_listbox_handlers import refresh_image_file_list
from cremage.ui.tool_palette import ToolPaletteArea
from cremage.ui.update_image_handler import update_image
from cremage.ui.face_ui_event_handlers import open_face_dir_button_handler
from cremage.ui.face_ui_event_handlers import copy_current_image_to_face_button_handler
from cremage.ui.show_prompt_list_handlers import show_positive_prompt_history_handler
from cremage.ui.show_prompt_list_handlers import show_negative_prompt_history_handler
from cremage.ui.show_prompt_list_handlers import show_positive_prompt_pre_expansion_history_handler
from cremage.ui.show_prompt_list_handlers import show_negative_prompt_pre_expansion_history_handler
from cremage.ui.show_prompt_list_handlers import show_positive_prompt_expansion_history_handler
from cremage.ui.show_prompt_list_handlers import show_negative_prompt_expansion_history_handler
from cremage.ui.copy_prompt_handlers import copy_positive_prompt_handler
from cremage.ui.copy_prompt_handlers import copy_negative_prompt_handler
from cremage.utils.sampler_utils import sampler_name_list
from cremage.utils.hires_fix_upscaler_utils import hires_fix_upscaler_name_list
from cremage.utils.misc_utils import join_directory_and_file_name
from cremage.ui.generator_model_type_change_handler import generator_model_type_changed
from cremage.ui.sdxl_use_refiner_check_box_handler import sdxl_use_refiner_check_box_changed
from cremage.ui.auto_face_fix_change_handler import auto_face_fix_changed
from cremage.utils.pixart_sigma_utils import pixart_sigma_model_id_cb_changed
from cremage.utils.pixart_sigma_utils import update_pixart_sigma_model_id_value
from sdxl.const.const import SDXL_RESOLUTIONS
from sdxl.const.const import SDXL_SAMPLER_NAME_LIST
from sdxl.const.const import DISCRETIZATION_LIST
from sdxl.const.const import GUIDER_LIST

BLANK_INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_input_image.png")
BLANK_MASK_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_mask_image.png")
BLANK_CONTROL_NET_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_image_control_net.png")
LIST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "prompt_history.png")
PROMPT_COPY_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "prompt_copy.png")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def _get_tool_processed_file_path_call_back(app):
    return os.path.join(app.output_dir, str(time.time()) + "_processed.png")

def _get_tool_processed_video_call_back(app):
    return os.path.join(app.output_dir, str(time.time()) + "_video.mp4")

def _generation_information_call_back(app):
    """Call back to get current generation information"""
    return app.current_image_generation_information_dict


def main_ui_definition(app) -> None:
    """
    Defines UI

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    def add_save_preferences_button(pos=(1, 0, 1, 1)):
        """
        Adds Save Preferences Button at the bottom of each tab.

        Arg
            pos: Grid position. Column number, row number, width, height.
        """
        hbox = Gtk.Grid()

        # Spacing between hbox elements
        hbox.set_row_spacing(10)
        hbox.set_column_spacing(10)
        vbox.pack_start(hbox, True, True, 0)

        save_button = Gtk.Button(label="Save as default")
        save_button.connect("clicked", lambda widget, app=app: save_preference_handler(app, widget))

        hbox.attach(save_button, 1, 0, 1, 1)
        hbox.set_halign(Gtk.Align.CENTER)
        hbox.set_valign(Gtk.Align.CENTER)

    # Container contains
    # * menu
    # * root box
    # Root box contains:
    # * box for image field, prompts
    # * box for image list
    # * box for preferences window
    app.set_default_size(1800, 1000)
    app.set_border_width(10)
    vbox_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    app.add(vbox_container)

    main_menu_definition(app)

    vbox_container.pack_start(
        app.menu_bar,
        False,  # Do not expand if the parent window expands
        True,   # Fill the space assigned by the container
        0)      # Padding on both sides of the menu

    # # Main layout container
    vbox_root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

    # # vbox on the left to store image listbox
    vbox_list = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

    ## Image listbox
    # current_image_start_index is the index of the file from the entire list
    # of the generated files that is currently on the image list view
    # e.g. if there are 1000 generated images, and if this index is 5,
    # then the 6th file should be displayed in the image view list.
    # if the image_per_page = 4, then index 5, 6, 7, 8 will be displayed on the list
    app.images_per_page = 6  # Number of images displayed on a list at a time
    app.current_image_start_index = 0

    # Control tab visibility on the right side of the UI
    # Key is the tab name
    app.tab_labels = dict()
    app.tab_contents = dict()

    # Update the image file list from the disk
    refresh_image_file_list(app, skip_adjustment=True)

    # Image listbox
    app.listbox = Gtk.ListBox()
    vbox_list.pack_start(app.listbox, False, True, 0)
    app.listbox.connect("key-press-event", lambda widget, event, app=app: image_listbox_key_press_handler(app, widget, event))
    app.listbox.connect("row-activated", lambda lb, row, app=app: on_row_activated(app, lb, row))

    # Scrollbar
    app.adjustment = Gtk.Adjustment(0, 0, app.total_positions, 1, 10, 0)
    app.scrollbar = Gtk.Scrollbar(orientation=Gtk.Orientation.VERTICAL, adjustment=app.adjustment)
    vbox_list.pack_start(app.scrollbar, False, True, 0)
    app.adjustment.connect("value-changed", lambda adjustment, app=app: on_scrollbar_value_changed(app, adjustment))

    # Initialize listbox
    update_listbox_all(app, 0)

    vbox_root.pack_start(vbox_list,
                        False, # Do not expand when parent window expands
                        True,  # Fill the initially assigned space
                        0)

    ## Right panel for the main image
    # Create a ScrolledWindow
    rootScrolledWindow = Gtk.ScrolledWindow()
    rootScrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC) # Show scrollbars as necessary

    # Add vbox_root to the ScrolledWindow
    rootScrolledWindow.add_with_viewport(vbox_root)

    # Set the minimum size of the scrolled window's content
    # Note that the user cannot make the window size smaller than this
    # Overall window size is set at the beginning of this method.
    # See app.set_default_size().
    rootScrolledWindow.set_min_content_width(1000)
    rootScrolledWindow.set_min_content_height(600)

    # Add the ScrolledWindow to vbox_container instead of adding vbox_root directly
    vbox_container.pack_start(rootScrolledWindow, True, True, 0)

    # vbox_container.pack_start(vbox_root, True, True, 0)
    vboxImage = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    vbox_root.pack_start(vboxImage, True, True, 0)

    # Image field
    # Wrap the input image in an event box to handle click events
    app.main_image_wrapper = Gtk.EventBox()

    img_placeholder = Image.new('RGBA', (MAIN_IMAGE_CANVAS_SIZE, MAIN_IMAGE_CANVAS_SIZE), "gray")
    app.image = Gtk.Image.new_from_pixbuf(
        pil_image_to_pixbuf(img_placeholder)
    )

    # Input image drag and drop support
    app.image.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
    app.image.drag_dest_add_text_targets()
    app.image.connect('drag-data-received',
                                       lambda
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                            app=app:
                                            main_image_drag_data_received(
                                            app,
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                        ))

    # Main image click support
    app.main_image_wrapper.add(app.image)
    # app.main_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
    app.main_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK)
    app.main_image_wrapper.connect("button-press-event", lambda widget, event, app=app: main_image_view_click_handler(app, widget, event))
    vboxImage.pack_start(app.main_image_wrapper, True, False, 0)

    ## Positive prompt fields
    # Label and button
    hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
    
    # Create and configure the label
    positive_prompt_label = Gtk.Label(label="Positive prompt")
    positive_prompt_label.set_halign(Gtk.Align.START)  # Align label to the left
    hbox.pack_start(positive_prompt_label, True, True, 0)

    # Create the button with an image
    icon_image = Gtk.Image.new_from_file(LIST_IMAGE_PATH)
    icon_button = Gtk.Button()
    icon_button.set_image(icon_image)
    icon_button.set_relief(Gtk.ReliefStyle.NONE)  # Optional: remove button border
    icon_button.connect("clicked", lambda widget, app=app: show_positive_prompt_history_handler(app, widget))
    hbox.pack_end(icon_button, False, False, 0)

    # Create the copy positive prompt button with an image
    icon_image = Gtk.Image.new_from_file(PROMPT_COPY_IMAGE_PATH)
    icon_button = Gtk.Button()
    icon_button.set_image(icon_image)
    icon_button.set_relief(Gtk.ReliefStyle.NONE)  # Optional: remove button border
    icon_button.connect("clicked", lambda widget, app=app: copy_positive_prompt_handler(app, widget))
    hbox.pack_end(icon_button, False, False, 0)

    vboxImage.pack_start(hbox, False, False, 0)
    # Positive prompt label end

    # Frame for the text view with a 1-pixel black border
    positive_frame = Gtk.Frame()
    positive_frame.set_shadow_type(Gtk.ShadowType.IN)  # Gives the impression of an inset (bordered) widget
    positive_frame.set_border_width(1)  # Adjust the positive_frame border width if necessary

    # Positive prompt multi-line text field
    app.positive_prompt = Gtk.TextView()
    app.positive_prompt.set_wrap_mode(Gtk.WrapMode.WORD)

    # Add the text view to the positive_frame, and then add the positive_frame to the vbox
    positive_frame.add(app.positive_prompt)
    vboxImage.pack_start(positive_frame, False, False, 0)

    ## Negative prompt fields
    # Label and button
    hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
    
    # Create and configure the label
    negative_prompt_label = Gtk.Label(label="Negative prompt")
    negative_prompt_label.set_halign(Gtk.Align.START)  # Align label to the left
    hbox.pack_start(negative_prompt_label, True, True, 0)

    # Create the button with an image
    icon_image = Gtk.Image.new_from_file(LIST_IMAGE_PATH)
    icon_button = Gtk.Button()
    icon_button.set_image(icon_image)
    icon_button.set_relief(Gtk.ReliefStyle.NONE)  # Optional: remove button border
    icon_button.connect("clicked", lambda widget, app=app: show_negative_prompt_history_handler(app, widget))
    hbox.pack_end(icon_button, False, False, 0)

    # Create the copy negative prompt button with an image
    icon_image = Gtk.Image.new_from_file(PROMPT_COPY_IMAGE_PATH)
    icon_button = Gtk.Button()
    icon_button.set_image(icon_image)
    icon_button.set_relief(Gtk.ReliefStyle.NONE)  # Optional: remove button border
    icon_button.connect("clicked", lambda widget, app=app: copy_negative_prompt_handler(app, widget))
    hbox.pack_end(icon_button, False, False, 0)

    vboxImage.pack_start(hbox, False, False, 0)
    # Negative prompt label end

    # Negative prompt multi-line text field
    # Frame for the text view with a 1-pixel black border
    negative_frame = Gtk.Frame()
    negative_frame.set_shadow_type(Gtk.ShadowType.IN)  # Gives the impression of an inset (bordered) widget
    negative_frame.set_border_width(1)  # Adjust the negative_frame border width if necessary

    # Negative multi-line text field
    app.negative_prompt = Gtk.TextView()
    app.negative_prompt.set_wrap_mode(Gtk.WrapMode.WORD)

    # Add the text view to the negative_frame, and then add the negative_frame to the vbox
    negative_frame.add(app.negative_prompt)
    vboxImage.pack_start(negative_frame, False, False, 0)

    # Generate button
    button_box = Gtk.Box(spacing=6)
    app.generate_button = Gtk.Button(label="Generate")
    button_box.set_halign(Gtk.Align.CENTER)  # Center the button horizontally
    button_box.pack_start(app.generate_button, False, False, 0)
    vboxImage.pack_start(button_box, False, False, 0)
    # app.generate_button.connect('button-press-event', generate_handler)
    app.generate_button.connect('button-press-event', lambda widget, event, app=app: generate_handler(app, widget, event))

    # Checkbox to use the generation info to override system preference
    app.override_checkbox = Gtk.CheckButton(label="Use generation info")
    # Commenting out as this is not needed for now
    # app.override_checkbox.connect("toggled", app.on_override_checkbox_toggled)
    button_box.pack_start(app.override_checkbox, False, False, 0)

    # Generation status
    app.generation_status = Gtk.Label(label="")
    app.generation_status.set_size_request(200, -1)  # Width = 30 pixels, height is default
    button_box.pack_start(app.generation_status, False, False, 0)  # widget, expand, fill, padding

    ## Generation information field
    # Label for the Positive Prompt text field
    generation_information_label = Gtk.Label()
    generation_information_label.set_text("Generation information")
    generation_information_label.set_halign(Gtk.Align.START)  # Align label to the left
    vboxImage.pack_start(generation_information_label, False, False, 0)

    # Frame for the text view with a 1-pixel black border
    generation_information_frame = Gtk.Frame()
    generation_information_frame.set_shadow_type(Gtk.ShadowType.IN)  # Gives the impression of an inset (bordered) widget
    generation_information_frame.set_border_width(1)  # Adjust the positive_frame border width if necessary

    # Generation_information multi-line text field
    app.generation_information = Gtk.TextView()
    app.generation_information.set_wrap_mode(Gtk.WrapMode.WORD)

    # Add the text view to the frame, and then add the frame to the vbox
    generation_information_frame.add(app.generation_information)
    vboxImage.pack_start(generation_information_frame, False, False, 0)

    ## Image input fields
    image_input_button_container_box = Gtk.Box(spacing=6)
    image_input_button_container_box.set_halign(Gtk.Align.CENTER)

    # Create the first radio button with its label
    app.rb_text_to_image = Gtk.RadioButton.new_with_label_from_widget(None, "Text to Image")
    app.rb_text_to_image.connect("toggled", lambda button, app=app: generation_mode_toggle_handler(app, button, "text to image"))
    image_input_button_container_box.pack_start(app.rb_text_to_image, False, False, 0)

    # Create the second radio button with its label
    app.rb_image_to_image = Gtk.RadioButton.new_with_label_from_widget(app.rb_text_to_image, "Image to Image")
    app.rb_image_to_image.connect("toggled", lambda button, app=app: generation_mode_toggle_handler(app, button, "image to image"))
    image_input_button_container_box.pack_start(app.rb_image_to_image, False, False, 0)

    # Create the third radio button with its label
    app.rb_inpainting = Gtk.RadioButton.new_with_label_from_widget(app.rb_text_to_image, "Inpainting")
    app.rb_inpainting.connect("toggled", lambda button, app=app: generation_mode_toggle_handler(app, button, "inpainting"))
    image_input_button_container_box.pack_start(app.rb_inpainting, False, False, 0)

    vboxImage.pack_start(image_input_button_container_box, False, False, 0)

    # Input image
    # Input images = (input image, mask image)
    # Hiearchy
    # vBoxImage
    #   input_images_container_box
    #     input_image_container_box
    #       image_input
    #     mask_image_container_box
    #       mask_image
    #
    # Container
    input_images_container_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    input_images_container_box.set_halign(Gtk.Align.CENTER)  # Center the button horizontally

    # Wrap the input image in an event box to handle click events
    app.input_image_wrapper = Gtk.EventBox()
    input_images_container_box.pack_start(app.input_image_wrapper, False, False, 0)

    # Input image
    app.image_input = resized_gtk_image_from_file(
        file_path=BLANK_INPUT_IMAGE_PATH,
        target_width=THUMBNAIL_IMAGE_EDGE_LENGTH,
        target_height=THUMBNAIL_IMAGE_EDGE_LENGTH)

    # Input image drag and drop support
    app.image_input.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
    app.image_input.drag_dest_add_text_targets()
    app.image_input.connect('drag-data-received',
                                       lambda
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                            app=app:
                                            input_image_drag_data_received(
                                            app,
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                        ))

    # Input image click support
    app.input_image_wrapper.add(app.image_input)
    app.input_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
    app.input_image_wrapper.connect("button-press-event", lambda widget, event, app=app: input_image_view_click_handler(app, widget, event))

    # Mask image
    # Wrap the image in an event box to handle click events
    app.mask_image_wrapper = Gtk.EventBox()
    input_images_container_box.pack_start(app.mask_image_wrapper, False, False, 0)

    # Create the mask image, initially gray
    app.mask_image = resized_gtk_image_from_file(
        file_path=BLANK_MASK_IMAGE_PATH,
        target_width=THUMBNAIL_IMAGE_EDGE_LENGTH,
        target_height=THUMBNAIL_IMAGE_EDGE_LENGTH)
    app.mask_image_wrapper.add(app.mask_image)

    # Connect the event box click event
    app.mask_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
    app.mask_image_wrapper.connect("button-press-event", lambda widget, event, app=app: mask_image_view_click_handler(app, widget, event))
    input_images_container_box.pack_start(app.mask_image_wrapper, False, False, 0)

    vboxImage.pack_start(input_images_container_box, True, False, 0)
    # Note that img_input is hidden in window_realize method handler

    # Copy image from output button
    copy_image_button_box = Gtk.Box(spacing=6)
    app.copy_image_button = Gtk.Button(label="Copy from main image area")
    copy_image_button_box.set_halign(Gtk.Align.CENTER)  # Center the button horizontally
    copy_image_button_box.pack_start(app.copy_image_button, False, False, 0)
    vboxImage.pack_start(copy_image_button_box, False, False, 0)
    app.copy_image_button.connect('button-press-event', lambda widget, event, app=app:copy_image_widget_click_handler(app, widget, event))

    # Preferences panel on the right
    update_ldm_model_name_value_from_ldm_model_dir(app)
    update_ldm_inpaint_model_name_value_from_ldm_model_dir(app)
    update_vae_model_name_value_from_vae_model_dir(app)
    update_control_model_name_value_from_control_model_dir(app)
    update_lora_model_name_value_from_lora_model_dir(app)

    update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir(app)
    update_sdxl_ldm_inpaint_model_name_value_from_sdxl_ldm_model_dir(app)
    update_sdxl_vae_model_name_value_from_sdxl_vae_model_dir(app)
    update_sdxl_lora_model_name_value_from_sdxl_lora_model_dir(app)

    update_pixart_sigma_ldm_model_name_value_from_pixart_sigma_ldm_model_dir(app)

    # Notebook for the tabbed UI for the right panel
    # Page 1 hierarchy:
    # vbox_root
    #   app.notebook
    #     tab
    #       vbox
    #         grid
    app.notebook = Gtk.Notebook()
    vbox_root.pack_start(app.notebook, True, True, 0)

    # Page 1 start
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    # vbox_root.pack_start(vbox, True, True, 0)
    app.notebook.append_page(vbox, Gtk.Label(label="Basic"))

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)

    # Create fields
    # app.fields contain fields that hold user preference values
    # with some exceptions

    # CSS Styling
    css = b"""
    textview text {
        border: 1px solid #818181; /* Light gray border */
        border-radius: 5px; /* Rounded corners */
        padding: 30px;
    }
    """
    css_provider = Gtk.CssProvider()
    css_provider.load_from_data(css)

    generator_model_type_list = GENERATOR_MODEL_TYPE_LIST
    sampler_list = list(filter(lambda e: e.find("DDIM") >= 0, sampler_name_list)) if app.preferences["hide_k_diffusion_samplers"] else sampler_name_list  # If you need to filter out any sampler, do here.
    sdxl_sampler_list = SDXL_SAMPLER_NAME_LIST
    hires_fix_upscaler_list = hires_fix_upscaler_name_list
    
    sdxl_resolution_str_list = [f"{w}x{h}" for (w, h) in SDXL_RESOLUTIONS]
    # Try to see if exact match is found if SD resolution is stored in preference.
    # If not, fall back to 1024x1024.
    try:
        resolution_index =  sdxl_resolution_str_list.index(
            f'{app.preferences["image_width"]}x{app.preferences["image_height"]}')
    except:
        resolution_index =  sdxl_resolution_str_list.index("1024x1024")

    def create_list_button(handler):
        icon_image = Gtk.Image.new_from_file(LIST_IMAGE_PATH)
        icon_button = Gtk.Button()
        icon_button.set_image(icon_image)
        icon_button.set_relief(Gtk.ReliefStyle.NONE)  # Optional: remove button border
        # icon_button.connect("clicked", lambda widget, app=app: show_positive_prompt_history_handler(app, widget))
        icon_button.connect("clicked", lambda widget, app=app: handler(app, widget))
        return icon_button
    
    fields1 = {
        "generator_model_type": create_combo_box_typeahead(generator_model_type_list, generator_model_type_list.index(app.preferences["generator_model_type"])),
        "sampling_steps": Gtk.Entry(text=str(app.preferences["sampling_steps"])),
        "sampler": create_combo_box_typeahead(sampler_list, sampler_list.index(app.preferences["sampler"])),
        "sdxl_sampler": create_combo_box_typeahead(sdxl_sampler_list, sdxl_sampler_list.index(app.preferences["sdxl_sampler"])),
        "image_width": Gtk.Entry(text=str(app.preferences["image_width"])),
        "image_height": Gtk.Entry(text=str(app.preferences["image_height"])),
        "sdxl_image_resolution":  create_combo_box_typeahead(sdxl_resolution_str_list, resolution_index),
        "clip_skip": Gtk.Entry(text=str(app.preferences["clip_skip"])),
        "denoising_strength": Gtk.Entry(text=str(app.preferences["denoising_strength"])),
        "batch_size": Gtk.Entry(text=str(app.preferences["batch_size"])),
        "number_of_batches": Gtk.Entry(text=str(app.preferences["number_of_batches"])),
        "positive_prompt_pre_expansion": Gtk.TextView(),
        "positive_prompt_pre_expansion_history": create_list_button(show_positive_prompt_pre_expansion_history_handler),
        "negative_prompt_pre_expansion": Gtk.TextView(),
        "negative_prompt_pre_expansion_history": create_list_button(show_negative_prompt_pre_expansion_history_handler),
        "positive_prompt_expansion": Gtk.TextView(),
        "positive_prompt_expansion_history": create_list_button(show_positive_prompt_expansion_history_handler),
        "negative_prompt_expansion": Gtk.TextView(),
        "negative_prompt_expansion_history": create_list_button(show_negative_prompt_expansion_history_handler),
        "enable_positive_prompt_pre_expansion": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["enable_positive_prompt_pre_expansion"])),
        "enable_negative_prompt_pre_expansion": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["enable_negative_prompt_pre_expansion"])),
        "enable_positive_prompt_expansion": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["enable_positive_prompt_expansion"])),
        "enable_negative_prompt_expansion": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["enable_negative_prompt_expansion"])),
        "seed": Gtk.Entry(text=str(app.preferences["seed"])),
        "cfg": Gtk.Entry(text=str(app.preferences["cfg"])),
        "hires_fix_upscaler": create_combo_box_typeahead(hires_fix_upscaler_list, hires_fix_upscaler_list.index(app.preferences["hires_fix_upscaler"])),
        "hires_fix_scale_factor": Gtk.Entry(text=str(app.preferences["hires_fix_scale_factor"])),
        "low_mem": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["low_mem"])),
        "keep_instance": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["keep_instance"])),
        "auto_face_fix": create_combo_box(TRUE_FALSE_LIST, int(not app.preferences["auto_face_fix"])),
        "auto_face_fix_face_detection_method": create_combo_box(FACE_DETECTION_METHOD_LIST, int(not app.preferences["auto_face_fix_face_detection_method"])),
        "auto_face_fix_strength": Gtk.Entry(text=app.preferences["auto_face_fix_strength"]),
        "auto_face_fix_prompt": Gtk.TextView(),
    }

    fields1["generator_model_type"].connect("changed", lambda widget, app=app:generator_model_type_changed(app, widget))

    text_view_set_text(fields1["positive_prompt_pre_expansion"], app.preferences["positive_prompt_pre_expansion"])
    text_view_set_text(fields1["negative_prompt_pre_expansion"], app.preferences["negative_prompt_pre_expansion"])
    text_view_set_text(fields1["positive_prompt_expansion"], app.preferences["positive_prompt_expansion"])
    text_view_set_text(fields1["negative_prompt_expansion"], app.preferences["negative_prompt_expansion"])

    fields1["auto_face_fix"].connect("changed", lambda widget, app=app:auto_face_fix_changed(app, widget))
    text_view_set_text(fields1["auto_face_fix_prompt"], app.preferences["auto_face_fix_prompt"])

    # Apply the same style to both TextViews
    for textview in [
        fields1["positive_prompt_pre_expansion"],
        fields1["negative_prompt_pre_expansion"],
        fields1["positive_prompt_expansion"],
        fields1["negative_prompt_expansion"],
        fields1["auto_face_fix_prompt"],
        ]:
        context = textview.get_style_context()
        context.add_provider(css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        textview.set_left_margin(10)  # Sets the left margin
        textview.set_right_margin(10)  # Sets the right margin
        textview.set_top_margin(10)  # Sets the top margin
        textview.set_bottom_margin(10)  # Sets the bottom margin

        # Enable word wrapping.
        # If you don't do this, textviews would make the parent window
        # extend too wide.
        textview.set_wrap_mode(Gtk.WrapMode.WORD)

    # value: ((label grid pos), (value field grid pos))
    row = 0
    fields1_pos = OrderedDict()
    fields1_pos["generator_model_type"] = ((0, row, 1, 1), (1, row, 1, 1))
    row += 1

    # Place both samplers at the Same position as display is toggled
    # by the generator type combobox
    fields1_pos["sampling_steps"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["sampler"] = ((2, row, 1, 1), (3, row, 1, 1))
    fields1_pos["sdxl_sampler"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["image_width"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["image_height"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["sdxl_image_resolution"] = ((0, row, 1, 1), (1, row, 1, 1))
    row += 1

    fields1_pos["batch_size"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["number_of_batches"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["positive_prompt_pre_expansion"] = ((0, row, 1, 1), (1, row, 3, 2))
    fields1_pos["positive_prompt_pre_expansion_history"] = ((4, row, 1, 1), (4, row, 1, 1))
    row += 2

    fields1_pos["negative_prompt_pre_expansion"] = ((0, row, 1, 1), (1, row, 3, 2))
    fields1_pos["negative_prompt_pre_expansion_history"] = ((4, row, 1, 1), (4, row, 1, 1))
    row += 2

    fields1_pos["enable_positive_prompt_pre_expansion"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["enable_negative_prompt_pre_expansion"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["positive_prompt_expansion"] = ((0, row, 1, 1), (1, row, 3, 2))
    fields1_pos["positive_prompt_expansion_history"] = ((4, row, 1, 1), (4, row, 1, 1))
    row += 2

    fields1_pos["negative_prompt_expansion"] = ((0, row, 1, 1), (1, row, 3, 2))
    fields1_pos["negative_prompt_expansion_history"] = ((4, row, 1, 1), (4, row, 1, 1))
    row += 2

    fields1_pos["enable_positive_prompt_expansion"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["enable_negative_prompt_expansion"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["seed"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["cfg"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["hires_fix_upscaler"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["hires_fix_scale_factor"] = ((2, row, 1, 1), (3, row, 1, 1))    
    row += 1

    fields1_pos["denoising_strength"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["clip_skip"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["low_mem"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["keep_instance"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["auto_face_fix"] = ((0, row, 1, 1), (1, row, 1, 1))
    fields1_pos["auto_face_fix_strength"] = ((2, row, 1, 1), (3, row, 1, 1))
    row += 1

    fields1_pos["auto_face_fix_face_detection_method"] = ((0, row, 1, 1), (1, row, 1, 1))
    row += 1

    fields1_pos["auto_face_fix_prompt"] = ((0, row, 1, 1), (1, row, 3, 2))
    row += 1

    assert sorted(fields1.keys()) == sorted(fields1_pos.keys())

    app.fields1_labels = dict()
    for k, v in fields1_pos.items():
        label = k.replace("_", " ").capitalize()
        app.fields1_labels[label] = Gtk.Label(
                label=label,
                halign=Gtk.Align.START)
        if v[0] != v[1]:  # Skip icon
            grid.attach(
                app.fields1_labels[label],
                *v[0])
        grid.attach(fields1[k], *v[1])

    # Save preferences button
    add_save_preferences_button()
    # end of page 1

    # Page 2
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    # vbox_root.pack_start(vbox, True, True, 0)
    # app.notebook.append_page(vbox, Gtk.Label(label="Models"))
    # app.tab_labels["Models"] = Gtk.Label(label="Models")
    # app.notebook.append_page(vbox, app.tab_labels["Models"])

    app.tab_labels["Models"] = Gtk.Label(label="Models")
    app.tab_contents["Models"] = vbox

    app.notebook.append_page(
        app.tab_contents["Models"],
        app.tab_labels["Models"])

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)
    fields2 = {
        "ldm_model": create_combo_box_typeahead(app.ldm_model_names, app.ldm_model_names.index(app.preferences["ldm_model"])),
        "ldm_inpaint_model": create_combo_box_typeahead(app.ldm_inpaint_model_names, app.ldm_inpaint_model_names.index(app.preferences["ldm_inpaint_model"])),
        "vae_model": create_combo_box_typeahead(app.vae_model_names, app.vae_model_names.index(app.preferences["vae_model"])),
        "lora_model_1": create_combo_box_typeahead(app.lora_model_names, app.lora_model_names.index(app.preferences["lora_model_1"])),
        "lora_weight_1": Gtk.Entry(text=app.preferences["lora_weight_1"]),
        "lora_model_2": create_combo_box_typeahead(app.lora_model_names, app.lora_model_names.index(app.preferences["lora_model_2"])),
        "lora_weight_2": Gtk.Entry(text=app.preferences["lora_weight_2"]),
        "lora_model_3": create_combo_box_typeahead(app.lora_model_names, app.lora_model_names.index(app.preferences["lora_model_3"])),
        "lora_weight_3": Gtk.Entry(text=app.preferences["lora_weight_3"]),
        "lora_model_4": create_combo_box_typeahead(app.lora_model_names, app.lora_model_names.index(app.preferences["lora_model_4"])),
        "lora_weight_4": Gtk.Entry(text=app.preferences["lora_weight_4"]),
        "lora_model_5": create_combo_box_typeahead(app.lora_model_names, app.lora_model_names.index(app.preferences["lora_model_5"])),
        "lora_weight_5": Gtk.Entry(text=app.preferences["lora_weight_5"]),
    }

    # Add fields to the grid
    row = 0
    for _, (label_text, field) in enumerate(fields2.items()):
        i = row
        if label_text.startswith("lora_weight") is False:
            label = Gtk.Label(label=label_text.replace("_", " ").capitalize(), halign=Gtk.Align.START)

            if re.search("lora_model_[\d+]", label_text):
                grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
                grid.attach(field, 1, i, 1, 1)
            else:  # Non-LoRA
                grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
                grid.attach(field, 1, i, 2, 1)
                row += 1
        else:  # LoRA weight
            grid.attach(field, 2, i, 1, 1)
            row += 1

    # Merge two dicts:
    # app.fields = {**fields1, **fields2}

    # Save preferences button
    add_save_preferences_button()
    # end of page 2

    # Page 2 for SDXL
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    # vbox_root.pack_start(vbox, True, True, 0)
    # app.notebook.append_page(vbox, Gtk.Label(label="SDXL Models"))
    # app.tab_labels["SDXL Models"] = app.notebook.get_tab_label(vbox)
    app.tab_labels["SDXL Models"] = Gtk.Label(label="SDXL Models")
    app.tab_contents["SDXL Models"] = vbox

    app.notebook.append_page(
        app.tab_contents["SDXL Models"],
        app.tab_labels["SDXL Models"])

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)
    fields2sdxl = {
        "sdxl_ldm_model": create_combo_box_typeahead(app.sdxl_ldm_model_names, app.sdxl_ldm_model_names.index(app.preferences["sdxl_ldm_model"])),
        "sdxl_ldm_inpaint_model": create_combo_box_typeahead(app.sdxl_ldm_inpaint_model_names, app.sdxl_ldm_inpaint_model_names.index(app.preferences["sdxl_ldm_inpaint_model"])),
        "sdxl_vae_model": create_combo_box_typeahead(app.sdxl_vae_model_names, app.sdxl_vae_model_names.index(app.preferences["sdxl_vae_model"])),
        "sdxl_lora_model_1": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["sdxl_lora_model_1"])),
        "sdxl_lora_weight_1": Gtk.Entry(text=app.preferences["sdxl_lora_weight_1"]),
        "sdxl_lora_model_2": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["sdxl_lora_model_2"])),
        "sdxl_lora_weight_2": Gtk.Entry(text=app.preferences["sdxl_lora_weight_2"]),
        "sdxl_lora_model_3": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["sdxl_lora_model_3"])),
        "sdxl_lora_weight_3": Gtk.Entry(text=app.preferences["sdxl_lora_weight_3"]),
        "sdxl_lora_model_4": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["sdxl_lora_model_4"])),
        "sdxl_lora_weight_4": Gtk.Entry(text=app.preferences["sdxl_lora_weight_4"]),
        "sdxl_lora_model_5": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["sdxl_lora_model_5"])),
        "sdxl_lora_weight_5": Gtk.Entry(text=app.preferences["sdxl_lora_weight_5"]),

        "sdxl_use_refiner": Gtk.CheckButton(label="Use refiner"),
        "sdxl_refiner_strength": Gtk.Entry(text=app.preferences["sdxl_refiner_strength"]),

        "refiner_sdxl_ldm_model": create_combo_box_typeahead(app.sdxl_ldm_model_names, app.sdxl_ldm_model_names.index(app.preferences["refiner_sdxl_ldm_model"])),
        "refiner_sdxl_vae_model": create_combo_box_typeahead(app.sdxl_vae_model_names, app.sdxl_vae_model_names.index(app.preferences["refiner_sdxl_vae_model"])),
        "refiner_sdxl_lora_model_1": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["refiner_sdxl_lora_model_1"])),
        "refiner_sdxl_lora_weight_1": Gtk.Entry(text=app.preferences["refiner_sdxl_lora_weight_1"]),
        "refiner_sdxl_lora_model_2": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["refiner_sdxl_lora_model_2"])),
        "refiner_sdxl_lora_weight_2": Gtk.Entry(text=app.preferences["refiner_sdxl_lora_weight_2"]),
        "refiner_sdxl_lora_model_3": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["refiner_sdxl_lora_model_3"])),
        "refiner_sdxl_lora_weight_3": Gtk.Entry(text=app.preferences["refiner_sdxl_lora_weight_3"]),
        "refiner_sdxl_lora_model_4": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["refiner_sdxl_lora_model_4"])),
        "refiner_sdxl_lora_weight_4": Gtk.Entry(text=app.preferences["refiner_sdxl_lora_weight_4"]),
        "refiner_sdxl_lora_model_5": create_combo_box_typeahead(app.sdxl_lora_model_names, app.sdxl_lora_model_names.index(app.preferences["refiner_sdxl_lora_model_5"])),
        "refiner_sdxl_lora_weight_5": Gtk.Entry(text=app.preferences["refiner_sdxl_lora_weight_5"]),
    }

    fields2sdxl["sdxl_use_refiner"].set_active(app.preferences["sdxl_use_refiner"])
    fields2sdxl["sdxl_use_refiner"].connect("toggled", lambda widget, app=app:sdxl_use_refiner_check_box_changed(app, widget))

    def sdxl_use_refiner_handler(widget):
        model_name = widget.get_child().get_text()
        if model_name.startswith("sd_xl_refiner"):
            show_error_dialog(app, """A standard refiner model (sd_xl_refiner) does not support LoRA.
                              Use a regular SDXL model to use LoRA.""")
    fields2sdxl["refiner_sdxl_ldm_model"].connect("changed", sdxl_use_refiner_handler)

    app.fields2_labels = dict()
    # Add fields to the grid
    row = 0
    for _, (label_text, field) in enumerate(fields2sdxl.items()):
        i = row
        if label_text.find("sdxl_lora_weight") < 0:
            if label_text == "sdxl_use_refiner":
                grid.attach(field, 0, i, 1, 1)  # field, left, top, width, height
            elif label_text == "sdxl_refiner_strength":
                label_text = label_text.replace("sdxl_", "").replace("_", " ").capitalize()
                label = Gtk.Label(label=label_text, halign=Gtk.Align.START)
                app.fields2_labels[label_text] = label
                grid.attach(label, 1, i, 1, 1)  # field, left, top, width, height
                grid.attach(field, 2, i, 1, 1)  # field, left, top, width, height
                row += 1
            else:
                label = Gtk.Label(label=label_text.replace("_", " ").capitalize(), halign=Gtk.Align.START)
                app.fields2_labels[label_text] = label

                if re.search("sdxl_lora_model_[\d+]", label_text):
                    grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
                    grid.attach(field, 1, i, 1, 1)
                else:  # Non-LoRA
                    grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
                    grid.attach(field, 1, i, 2, 1)
                    row += 1
        else:  # LoRA weight
            grid.attach(field, 2, i, 1, 1)
            row += 1

    # Merge two dicts:
    app.fields = {**fields1, **fields2, **fields2sdxl}

    # Save preferences button
    add_save_preferences_button()
    # end of page 2 model sdxl

    # Page 3
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    app.tab_labels["ControlNet"] = Gtk.Label(label="ControlNet")
    app.tab_contents["ControlNet"] = vbox
    app.notebook.append_page(
        app.tab_contents["ControlNet"],
        app.tab_labels["ControlNet"])

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)

    # ControlNet type
    label = Gtk.Label(label="ControlNet type")
    grid.attach(label, 0, 0, 1, 1)

    app.fields["control_model"] = create_combo_box_typeahead(
        app.control_model_names,
        app.control_model_names.index(app.preferences["control_model"]))
    grid.attach(app.fields["control_model"], 1, 0, 1, 1)

    # Control image label
    label = Gtk.Label(label="Control image")
    grid.attach(label, 0, 1, 1, 1)

    # Control image view
    # Create a wrapper to capture the button click on the image field
    # Otherwise, click won't trigger the on click event
    app.control_net_image_container_box = Gtk.EventBox()  # Note EventBox not Box
    app.control_net_image_view = resized_gtk_image_from_file(
        file_path=BLANK_CONTROL_NET_IMAGE_PATH,
        target_width=THUMBNAIL_IMAGE_EDGE_LENGTH,
        target_height=THUMBNAIL_IMAGE_EDGE_LENGTH)
    # Setup drag and drop for the image area
    app.control_net_image_view.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
    app.control_net_image_view.drag_dest_add_text_targets()
    app.control_net_image_view.connect('drag-data-received',
                                       lambda
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                            app=app:
                                        control_image_drag_data_received(
                                            app,
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                        ))

    # ControlNet image wrapper box
    app.control_net_image_container_box.add(app.control_net_image_view)
    app.control_net_image_container_box.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
    app.control_net_image_container_box.connect('button-press-event', lambda widget, event, app=app:control_image_view_click_handler(app, widget, event))
    grid.attach(app.control_net_image_container_box, 0, 2, 4, 4)

    # Graffiti editor button
    graffiti_editor_button = Gtk.Button(label="Graffiti editor")
    graffiti_editor_button.connect('button-press-event', lambda widget, event, app=app: graffiti_editor_widget_click_handler(app, widget, event))

    grid.attach(graffiti_editor_button, 1, 6, 2, 1)

    # Save preferences button
    add_save_preferences_button()
    # End of page 3

    # Start of page 4
    update_image_wrapper = partial(update_image, app)
    get_tool_processed_file_path_call_back_wrapper = partial(_get_tool_processed_file_path_call_back, app)
    get_tool_processed_video_call_back_wrapper = partial(_get_tool_processed_video_call_back, app)

    generation_information_call_back_wrapper = partial(_generation_information_call_back, app)

    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    app.notebook.append_page(vbox, Gtk.Label(label="Tools"))
    tools_area = ToolPaletteArea(vbox,
        get_current_image_call_back=app.get_current_image_call_back,
        get_tool_processed_file_path_call_back=get_tool_processed_file_path_call_back_wrapper,
        get_tool_processed_video_call_back=get_tool_processed_video_call_back_wrapper,
        save_call_back=update_image_wrapper,
        generation_information_call_back=generation_information_call_back_wrapper,
        preferences=app.preferences,
        positive_prompt=text_view_get_text(app.positive_prompt),
        negative_prompt=text_view_get_text(app.negative_prompt),
        get_current_face_image_call_back=app.get_current_face_image_call_back,
        face_model_full_path=join_directory_and_file_name(
                app.preferences["control_model_path"], FACE_MODEL_NAME),
        app=app
    )

    # Page 5
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    app.notebook.append_page(vbox, Gtk.Label(label="Face"))

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom
    row = 6

    vbox.pack_start(grid, True, True, 0)

    # Image selector
    # Wrap the input image in an event box to handle click events
    app.face_input_image_closable_container = Gtk.Box()
    app.face_input_image_closable_container.set_orientation(Gtk.Orientation.VERTICAL)
    app.face_input_image_wrapper = Gtk.EventBox()
    app.face_image_input = resized_gtk_image_from_file(
        file_path=BLANK_INPUT_IMAGE_PATH,
        target_width=THUMBNAIL_IMAGE_EDGE_LENGTH,
        target_height=THUMBNAIL_IMAGE_EDGE_LENGTH)

    # Input image drag and drop support
    app.face_image_input.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
    app.face_image_input.drag_dest_add_text_targets()
    app.face_image_input.connect('drag-data-received',
                                       lambda
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                            app=app:
                                            face_input_image_drag_data_received(
                                            app,
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                        ))    
    app.face_input_image_wrapper.add(app.face_image_input)
    app.face_input_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
    app.face_input_image_wrapper.connect("button-press-event", lambda widget, event, app=app: face_input_image_view_click_handler(app, widget, event))
    app.face_input_image_closable_container.pack_start(app.face_input_image_wrapper, True, True, 0)
    close_button = Gtk.Button(label="X")
    close_button.set_size_request(25, 25)  # Width, Height in pixels

    close_button.connect("clicked", lambda widget, app=app: face_input_image_view_close_handler(app, widget))
    app.face_input_image_closable_container.pack_start(close_button, False, False, 0)
    grid.attach(app.face_input_image_closable_container, 1, row, 2, 1)
    row += 1

    # Checkbox to disable face id
    app.disable_face_input_checkbox = Gtk.CheckButton(label="Disable face input")
    grid.attach(app.disable_face_input_checkbox, 1, row, 1, 1)
    row += 1

    # Strength
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
    box.set_halign(Gtk.Align.CENTER)
    box.set_valign(Gtk.Align.CENTER)
    
    label = Gtk.Label(label="Strength")
    box.pack_start(label, True, False, 0)
    app.fields["face_strength"] = Gtk.Entry(text="1.0")
    box.pack_start(app.fields["face_strength"], True, False, 0)
    grid.attach(box, 1, row, 1, 1)
    row += 1

    # Button to open face data directory
    app.open_face_dir_button = Gtk.Button(label="Open face image directory")
    app.open_face_dir_button.connect('button-press-event', lambda widget, event, app=app: open_face_dir_button_handler(app, widget, event))
    grid.attach(app.open_face_dir_button, 1, row, 1, 1)
    row += 1

    # Button to copy current image
    app.copy_current_image_to_face_button = Gtk.Button(label="Copy from current image")
    app.copy_current_image_to_face_button.connect('button-press-event', lambda widget, event, app=app: copy_current_image_to_face_button_handler(app, widget, event))
    grid.attach(app.copy_current_image_to_face_button, 1, row, 1, 1)
    row += 1

    # Save preferences button
    add_save_preferences_button()
    # End of page 5

    # Page 6 start: SDXL advanced
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    app.tab_labels["SDXL Advanced"] = Gtk.Label(label="SDXL Advanced")
    app.tab_contents["SDXL Advanced"] = vbox
    app.notebook.append_page(vbox, app.tab_labels["SDXL Advanced"])

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)
    # Create fields
    # app.fields contain fields that hold user preference values
    # with some exceptions

    # CSS Styling
    css = b"""
    textview text {
        border: 1px solid #818181; /* Light gray border */
        border-radius: 5px; /* Rounded corners */
        padding: 30px;
    }
    """
    css_provider = Gtk.CssProvider()
    css_provider.load_from_data(css)

    guider_list = list(filter(lambda e: e=="VanillaCFG", GUIDER_LIST))
    discretization_list = DISCRETIZATION_LIST

    fields_sdxl_advanced = {
        "discretization": create_combo_box_typeahead(discretization_list, discretization_list.index(app.preferences["discretization"])),
        "discretization_sigma_min": Gtk.Entry(text=str(app.preferences["discretization_sigma_min"])),
        "discretization_sigma_max": Gtk.Entry(text=str(app.preferences["discretization_sigma_max"])),
        "discretization_rho": Gtk.Entry(text=str(app.preferences["discretization_rho"])),
        "guider": create_combo_box_typeahead(guider_list, guider_list.index(app.preferences["guider"])),
        "linear_prediction_guider_min_scale": Gtk.Entry(text=str(app.preferences["linear_prediction_guider_min_scale"])),
        "linear_prediction_guider_max_scale": Gtk.Entry(text=str(app.preferences["linear_prediction_guider_max_scale"])),
        "triangle_prediction_guider_min_scale": Gtk.Entry(text=str(app.preferences["triangle_prediction_guider_min_scale"])),
        "triangle_prediction_guider_max_scale": Gtk.Entry(text=str(app.preferences["triangle_prediction_guider_max_scale"])),
        "sampler_s_churn": Gtk.Entry(text=str(app.preferences["sampler_s_churn"])),
        "sampler_s_tmin": Gtk.Entry(text=str(app.preferences["sampler_s_tmin"])),
        "sampler_s_tmax": Gtk.Entry(text=str(app.preferences["sampler_s_tmax"])),
        "sampler_s_noise": Gtk.Entry(text=str(app.preferences["sampler_s_noise"])),
        "sampler_eta": Gtk.Entry(text=str(app.preferences["sampler_eta"])),
        "sampler_order": Gtk.Entry(text=str(app.preferences["sampler_order"]))
    }

    # value: ((label grid pos), (value field grid pos))
    row = 0
    fields_sdxl_advanced_pos = OrderedDict()
    for k, w in fields_sdxl_advanced.items():
        fields_sdxl_advanced_pos[k] = ((0, row, 1, 1), (1, row, 1, 1))
        row += 1

    assert sorted(fields_sdxl_advanced.keys()) == sorted(fields_sdxl_advanced_pos.keys())

    app.fields_sdxl_advanced_labels = dict()
    for k, v in fields_sdxl_advanced_pos.items():
        label = k.replace("_", " ").capitalize()
        app.fields_sdxl_advanced_labels[label] = Gtk.Label(
                label=label,
                halign=Gtk.Align.START)
        grid.attach(app.fields_sdxl_advanced_labels[label], *v[0])
        grid.attach(fields_sdxl_advanced[k], *v[1])
    # Page 6 for SDXL advanced end

    # Page 7 for PixArt-Sigma
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    app.tab_labels["PixArt-Sigma"] = Gtk.Label(label="PixArt-Sigma")
    app.tab_contents["PixArt-Sigma"] = vbox

    app.notebook.append_page(
        app.tab_contents["PixArt-Sigma"],
        app.tab_labels["PixArt-Sigma"])

    grid = Gtk.Grid()
    grid.set_column_spacing(10)
    grid.set_row_spacing(10)

    # Set margins for the grid
    grid.set_margin_start(10)  # Margin on the left side
    grid.set_margin_end(10)    # Margin on the right side
    grid.set_margin_top(10)    # Margin on the top
    grid.set_margin_bottom(10) # Margin on the bottom

    vbox.pack_start(grid, True, True, 0)

    update_pixart_sigma_model_id_value(app)
    fields_pixart_sigma = {
        "pixart_sigma_model_id": create_combo_box_typeahead(
            app.pixart_sigma_model_ids,
            app.pixart_sigma_model_ids.index(app.preferences["pixart_sigma_model_id"]),
            width=400),
        "pixart_sigma_ldm_model": create_combo_box_typeahead(
            app.pixart_sigma_ldm_model_names,
            app.pixart_sigma_ldm_model_names.index(app.preferences["pixart_sigma_ldm_model"]),
            width=400)
    }
    fields_pixart_sigma["pixart_sigma_model_id"].connect("changed", lambda widget, app=app:pixart_sigma_model_id_cb_changed(app, widget))

    app.pixart_sigma_fields_labels = dict()
    # Add fields to the grid
    row = 0
    for _, (label_text, field) in enumerate(fields_pixart_sigma.items()):
        i = row

        label = Gtk.Label(label=label_text.replace("_", " ").capitalize(), halign=Gtk.Align.START)
        app.pixart_sigma_fields_labels[label_text] = label

        grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
        grid.attach(field, 1, i, 2, 1)
        row += 1

    # Save preferences button
    add_save_preferences_button()
    # end of page 7 pixart-sigma

    # Merge two dicts:
    tmp_fields = {**app.fields, **fields_pixart_sigma}
    app.fields = tmp_fields