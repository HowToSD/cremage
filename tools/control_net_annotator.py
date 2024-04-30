import os
import logging
import sys
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [PROJECT_ROOT, MODULE_ROOT, TOOLS_ROOT] + sys.path

from cremage.utils.image_utils import pil_image_to_pixbuf, resize_with_padding
from cremage.utils.gtk_utils import show_alert_dialog, resized_gtk_image_from_file
from cremage.control_net.annotator_wrapper import generate_canny
from cremage.control_net.annotator_wrapper import generate_depth_map
from cremage.control_net.annotator_wrapper import generate_open_pose
from cremage.control_net.annotator_wrapper import generate_fake_scribble
from cremage.control_net.annotator_wrapper import generate_scribble
from cremage.control_net.annotator_wrapper import generate_hed
from cremage.control_net.annotator_wrapper import generate_mlsd
from cremage.control_net.annotator_wrapper import generate_normal_map
from cremage.control_net.annotator_wrapper import generate_seg
from cremage.utils.gtk_utils import create_combo_box

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


BLANK_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "images", "blank_image.png")
OUTPUT_PLACEHOLDER_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "resources", "images", "output_placeholder.png")
ANNOTATOR_LIST = [
    "Canny",
    "Depth map",
    "OpenPose",
    "Fake scribble",
    "Scribble",
    "HED",
    "Hough (MLSD)",
    "Normal map",
    "Segmentation map"
]

class ControlNetImageAnnotator(Gtk.Window):
    def __init__(self, pil_image=None, output_file_path=None,
                 save_call_back=None):
        """
        Args:
            pil_image (Image): The input image in PIL format.
            output_file_path (str): The output path of the generated image.
            save_call_back: Call back to call when the output image is generated.
        """
        super().__init__(title="ControlNet Input Image Creator")
        self.pil_image = pil_image
        self.output_file_path = output_file_path
        self.save_call_back = save_call_back

        self.set_default_size(440, 400)  # w, h
        self.input_image_selected = False

        if self.pil_image is None:
            self.pil_image = Image.open(BLANK_IMAGE_PATH)

        self.output_pil_image = None
        
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create a MenuBar
        menubar = Gtk.MenuBar()
        vbox.pack_start(menubar, False, False, 0)

        # Create a File menu
        file_menu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(file_menu)
        menubar.append(file_item)

        # Add items to the File menu
        open_item = Gtk.MenuItem(label="Open")
        open_item.connect("activate", self.on_open_clicked)
        file_menu.append(open_item)

        save_item = Gtk.MenuItem(label="Save")
        save_item.connect("activate", self.on_save_clicked)
        file_menu.append(save_item)

        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", self.on_exit_clicked)
        file_menu.append(exit_item)

        file_menu.show_all()

        # Layout the rest of the UI 
        self.set_border_width(10)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        vbox.pack_start(grid, True, True, 0)

        # Adding widgets
        annotator_label = Gtk.Label(label="Control type")
        annotators = ANNOTATOR_LIST
        self.annotator_cb = create_combo_box(annotators, annotators.index("Canny"))
        self.annotator_cb.connect("changed", self.on_annotator_changed)

        self.canny_low_threshold_label = Gtk.Label(label="Canny low threshold")
        self.canny_low_threshold_entry = Gtk.Entry(text=str(100))
        self.canny_high_threshold_label = Gtk.Label(label="Canny high threshold")
        self.canny_high_threshold_entry = Gtk.Entry(text=str(200))

        self.mlsd_value_threshold_label = Gtk.Label(label="Hough value threshold")
        self.mlsd_value_threshold_entry = Gtk.Entry(text=str(0.1))  # [0.01, 2]
        self.mlsd_distance_threshold_label = Gtk.Label(label="Hough distance threshold")
        self.mlsd_distance_threshold_entry = Gtk.Entry(text=str(0.1)) # [0.01, 20]

        self.normal_background_threshold_label = Gtk.Label(label="Normal background threshold")
        self.normal_background_threshold_entry = Gtk.Entry(text=str(0.4))  # [0.0, 1.0]

        self.image1 = Gtk.Image.new_from_pixbuf(
            pil_image_to_pixbuf(
                resize_with_padding(self.pil_image, target_width=256,
                                    target_height=256)))
        # Setup drag and drop for the image area
        self.image1.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.image1.drag_dest_add_text_targets()
        self.image1.connect('drag-data-received', self.on_drag_data_received)
        
        # Output image
        self.image2 = resized_gtk_image_from_file(
            file_path=OUTPUT_PLACEHOLDER_IMAGE_PATH,
            target_width=256,
            target_height=256)
        self.generate_button = Gtk.Button(label="Generate")
        self.generate_button.connect("clicked", self.on_generate_clicked)
        
        # attach args: left, top, width, height
        grid.attach(annotator_label, 0, 0, 1, 1)
        grid.attach(self.annotator_cb, 1, 0, 1, 1)
        
        grid.attach(self.canny_low_threshold_label, 0, 1, 1, 1)
        grid.attach(self.canny_low_threshold_entry, 1, 1, 1, 1)

        grid.attach(self.canny_high_threshold_label, 2, 1, 1, 1)
        grid.attach(self.canny_high_threshold_entry, 3, 1, 1, 1)

        grid.attach(self.mlsd_value_threshold_label, 0, 2, 1, 1)
        grid.attach(self.mlsd_value_threshold_entry, 1, 2, 1, 1)

        grid.attach(self.mlsd_distance_threshold_label, 2, 2, 1, 1)
        grid.attach(self.mlsd_distance_threshold_entry, 3, 2, 1, 1)

        grid.attach(self.normal_background_threshold_label, 0, 3, 1, 1)
        grid.attach(self.normal_background_threshold_entry, 1, 3, 1, 1)

        grid.attach(self.image1, 0, 4, 2, 5)
        grid.attach(self.image2, 2, 4, 2, 5)

        grid.attach(self.generate_button, 4, 22, 1, 1)

        self.pil_image = None  # Placeholder for PIL input image

    def on_open_clicked(self, action):
        dialog = Gtk.FileChooserDialog(title="Please choose a file", parent=self, action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            image_path = dialog.get_filename()
            self.image1.set_from_file(image_path)
            # Update the input image
            self.pil_image = Image.open(image_path)
            self.input_image_selected = True
        dialog.destroy()

    def on_drag_data_received(self, widget, drag_context, x, y, data, info, time):
        """
        Drag and Drop handler.

        Args:
            data: An object that contains info for the dragged file name
        """
        file_path = data.get_text().strip()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
        logger.info(f"on_drag_data_received: {file_path}")
        self.pil_image = Image.open(file_path)
        resized_image = resize_with_padding(
            self.pil_image,
            target_width=256,
            target_height=256)
        self.image1.set_from_pixbuf(pil_image_to_pixbuf(resized_image))
        self.input_image_selected = True

    def on_save_clicked(self, action):
        if self.output_pil_image:
            dialog = Gtk.FileChooserDialog(title="Save File", parent=self, action=Gtk.FileChooserAction.SAVE)
            dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)

            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                save_path = dialog.get_filename()
                self.output_pil_image.save(save_path)
            dialog.destroy()

    def on_exit_clicked(self, action):
        Gtk.main_quit()

    def on_generate_clicked(self, action):
        if self.input_image_selected is False:
            show_alert_dialog("Select an input image.")
            return
        
        # self.pil_image is a valid input image.        
        w, h = self.pil_image.size
        resolution = min(w, h)

        if self.annotator_cb.get_active_text() == "Canny":
            self.output_pil_image = generate_canny(self.pil_image,
                        low_threshold=int(self.canny_low_threshold_entry.get_text()),
                        high_threshold=int(self.canny_high_threshold_entry.get_text()),
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Depth map":
            self.output_pil_image = generate_depth_map(self.pil_image,
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "OpenPose":
            self.output_pil_image = generate_open_pose(self.pil_image,
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Fake scribble":
            self.output_pil_image = generate_fake_scribble(self.pil_image,
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Scribble":
            self.output_pil_image = generate_scribble(self.pil_image,
                        image_resolution=resolution)            
        elif self.annotator_cb.get_active_text() == "HED":
            self.output_pil_image = generate_hed(self.pil_image,
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Hough (MLSD)":
            self.output_pil_image = generate_mlsd(self.pil_image,
                        float(self.mlsd_value_threshold_entry.get_text()),
                        float(self.mlsd_distance_threshold_entry.get_text()),
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Normal map":
            self.output_pil_image = generate_normal_map(self.pil_image,
                        float(self.normal_background_threshold_entry.get_text()),
                        image_resolution=resolution)
        elif self.annotator_cb.get_active_text() == "Segmentation map":
            self.output_pil_image = generate_seg(self.pil_image,
                         image_resolution=resolution)
        else:
            raise ValueError("Unsupported annotator")
        
        self.image2.set_from_pixbuf(pil_image_to_pixbuf(self.output_pil_image))
        
        if self.output_file_path:
            self.output_pil_image.save(self.output_file_path)

        if self.save_call_back:
            self.save_call_back(self.output_pil_image)
 
    def on_annotator_changed(self, combo):
        # Get the active text from the combo box
        model = combo.get_model()
        active_index = combo.get_active()
        if active_index < 0:
            return
        selected_annotator = model[active_index][0]

        # Toggle visibility based on the selection
        if selected_annotator == "Canny":
            self.canny_low_threshold_label.show()
            self.canny_low_threshold_entry.show()
            self.canny_high_threshold_label.show()
            self.canny_high_threshold_entry.show()
            self.mlsd_value_threshold_label.hide()
            self.mlsd_value_threshold_entry.hide()
            self.mlsd_distance_threshold_label.hide()
            self.mlsd_distance_threshold_entry.hide()
            self.normal_background_threshold_label.hide()
            self.normal_background_threshold_entry.hide()
        elif selected_annotator == "Hough (MLSD)":
            self.canny_low_threshold_label.hide()
            self.canny_low_threshold_entry.hide()
            self.canny_high_threshold_label.hide()
            self.canny_high_threshold_entry.hide()
            self.mlsd_value_threshold_label.show()
            self.mlsd_value_threshold_entry.show()
            self.mlsd_distance_threshold_label.show()
            self.mlsd_distance_threshold_entry.show()
            self.normal_background_threshold_label.hide()
            self.normal_background_threshold_entry.hide()
        elif selected_annotator == "Normal map":
            self.canny_low_threshold_label.hide()
            self.canny_low_threshold_entry.hide()
            self.canny_high_threshold_label.hide()
            self.canny_high_threshold_entry.hide()
            self.mlsd_value_threshold_label.hide()
            self.mlsd_value_threshold_entry.hide()
            self.mlsd_distance_threshold_label.hide()
            self.mlsd_distance_threshold_entry.hide()
            self.normal_background_threshold_label.show()
            self.normal_background_threshold_entry.show()
        else:
            # For other types or a default case
            self.canny_low_threshold_label.hide()
            self.canny_low_threshold_entry.hide()
            self.canny_high_threshold_label.hide()
            self.canny_high_threshold_entry.hide()
            self.mlsd_value_threshold_label.hide()
            self.mlsd_value_threshold_entry.hide()
            self.mlsd_distance_threshold_label.hide()
            self.mlsd_distance_threshold_entry.hide()
            self.normal_background_threshold_label.hide()
            self.normal_background_threshold_entry.hide()


if __name__ == "__main__":
    app = ControlNetImageAnnotator()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    app.on_annotator_changed(app.annotator_cb)
    Gtk.main()
