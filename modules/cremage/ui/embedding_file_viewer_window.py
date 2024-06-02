import os
import sys
import io
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_get_text, text_view_set_text, create_combo_box, show_error_dialog
from cremage.utils.ml_utils import load_torch_model_paths, load_embedding

SELECTION_MESSAGE = "Select the embedding file"


def pil_image_to_pixbuf(pil_img: Image.Image) -> GdkPixbuf.Pixbuf:
    """
    Converts a PIL Image object to a GdkPixbuf.Pixbuf object.

    This conversion allows the PIL Image to be used in GTK applications, 
    where GdkPixbuf is the standard format for images.

    Args:
        pil_img (PIL.Image.Image): The PIL Image object to convert.

    Returns:
        GdkPixbuf.Pixbuf: The resulting pixbuf object.

    Example:
        >>> from PIL import Image
        >>> pil_img = Image.open("example.png")
        >>> pixbuf = pil_image_to_pixbuf(pil_img)
    """
    # Save the PIL Image to a bytes buffer in PNG format
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)  # Seek to the start of the buffer
    
    # Load the image buffer into a GdkPixbuf
    loader = GdkPixbuf.PixbufLoader.new_with_type('png')
    loader.write(buf.getvalue())
    loader.close()
    pixbuf = loader.get_pixbuf()
    
    return pixbuf

class EmbeddingFileViewerWindow(Gtk.Window):

    def __init__(self, app=None, embedding_path=None, embedding_images_dir=None):
        super().__init__(title="TI Embedding Tool")
        self.app = app
        if self.app:
            self.positive_prompt_field = self.app.positive_prompt
            self.negative_prompt_field = self.app.negative_prompt
        else:
            self.positive_prompt_field = None
            self.negative_prompt_field = None
        self.embedding_dir_path = embedding_path
        self.embedding_images_dir = embedding_images_dir
        self.image_window = None
        self.selected_embedding_file_name = None
        self.selected_event_box = None
        self.set_border_width(10)
        
        vboxContainer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add(vboxContainer)
        # Hierarchy:
        # app_window
        #   vBoxContainer
        #     scrolled_window
        #       flowbox

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_min_content_width(150)
        scrolled_window.set_min_content_height(100)

        self.flowbox = Gtk.FlowBox()
        self.flowbox.set_max_children_per_line(0)
        self.flowbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.flowbox.connect("child-activated", self.on_image_clicked)
        scrolled_window.add(self.flowbox)
        
        vboxContainer.pack_start(scrolled_window, True, True, 0)

        self.set_default_size(800, 600)

        # Scan the embedding_images_directory and create a thumbnail
        # of each image file if following conditions are met:
        # 1) Thumbnail file is missing for the image file.
        # 2) Image file is newer than the thumbnail file.
        # In terms of naming, if the original file name is:
        #   foo.safetensors.png,
        # Thumbnail should be named:
        # foo.safetensors.thumbnail.png

        # If one or more thumbnail file need to be created,
        # Then display a dialog that says:
        #  "Creating thumbnails of images"
        # Close the dialog once thumbnails are created.
        # Do not display this dialog if no thumbnail file needs to be
        # created.

        self.create_thumbnails()
        self.load_images_and_files()

        # Embedding shape button
        self.embedding_shape_text = Gtk.TextView()
        self.embedding_shape_text.set_wrap_mode(Gtk.WrapMode.WORD)
        vboxContainer.pack_start(self.embedding_shape_text, False, True, 0)

        # Add to positive prompt button
        self.add_to_positive_prompt_button = Gtk.Button(label="Add to positive prompt")
        self.add_to_positive_prompt_button.connect("clicked", self.add_to_positive_prompt_button_pressed)
        vboxContainer.pack_start(self.add_to_positive_prompt_button, False, True, 0)
   
        # Add to negative prompt button
        self.add_to_negative_prompt_button = Gtk.Button(label="Add to negative prompt")
        self.add_to_negative_prompt_button.connect("clicked", self.add_to_negative_prompt_button_pressed)
        vboxContainer.pack_start(self.add_to_negative_prompt_button, False, True, 0)

        # Analyze button
        self.analyze_button = Gtk.Button(label="Show data shape")
        self.analyze_button.connect("clicked", self.analyze_pressed)
        vboxContainer.pack_start(self.analyze_button, False, True, 0)

    def create_thumbnails(self):
        thumbnails_to_create = []
        for file_name in os.listdir(self.embedding_images_dir):
            if file_name.endswith(".png") and not file_name.endswith(".thumbnail.png"):
                image_path = os.path.join(self.embedding_images_dir, file_name)
                thumbnail_path = os.path.join(self.embedding_images_dir, f"{os.path.splitext(file_name)[0]}.thumbnail.png")
                if not os.path.exists(thumbnail_path) or os.path.getmtime(image_path) > os.path.getmtime(thumbnail_path):
                    thumbnails_to_create.append((image_path, thumbnail_path))

        if thumbnails_to_create:
            dialog = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO, Gtk.ButtonsType.NONE, "Creating thumbnails of images")
            dialog.show()

            for image_path, thumbnail_path in thumbnails_to_create:
                image = Image.open(image_path)
                image.thumbnail((128, 128), Image.LANCZOS)
                image.save(thumbnail_path, format='PNG')

            dialog.destroy()

    def load_images_and_files(self):
        # Load and sort items case-insensitively
        embedding_file_paths = load_torch_model_paths(self.embedding_dir_path)
        items_sorted = sorted(embedding_file_paths, key=lambda s: s.lower())

        for item in items_sorted:
            image_path = os.path.join(self.embedding_images_dir, item + ".thumbnail.png")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                # image.thumbnail((128, 128), Image.LANCZOS)  # No need to do this
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_path, image.width, image.height, True)
            else:
                image = Image.new('RGB', (64, 96), color='gray')  # w, h
                # image.thumbnail((96, 128), Image.LANCZOS)
                pixbuf = pil_image_to_pixbuf(image)

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            img = Gtk.Image.new_from_pixbuf(pixbuf)
            label = Gtk.Label(label=item)
            
            # Hierarchy
            # flowbox
            #   event_box
            #     box
            #       image
            #       label
            event_box = Gtk.EventBox()
            event_box.add(box)

            box.pack_start(img, False, False, 0)
            box.pack_start(label, False, False, 0)

            self.flowbox.add(event_box)

        self.show_all()

    def on_image_clicked(self, flowbox, child):
        print("image clicked")
        for other_child in self.flowbox.get_children():
            other_child.get_style_context().remove_class("selected")
        child.get_style_context().add_class("selected")
        
        event_box = child.get_child()  # EventBox
        box = event_box.get_child()  # Box
        children = box.get_children() # image, label
        
        # Ensure there are enough children and get the label
        if len(children) > 1 and isinstance(children[1], Gtk.Label):
            label = children[1]
            self.selected_embedding_file_name = label.get_text()
            self.selected_event_box = child
        else:
            self.selected_embedding_file_name = None

    def show_image_window(self, embedding_file_name):
        image_path = os.path.join(self.embedding_images_dir, embedding_file_name + ".png")
        if os.path.exists(image_path):
            if self.image_window is not None:
                self.image_window.destroy()
            
            image = Image.open(image_path)
            width, height = image.size
            if width > height:
                new_width = 768
                new_height = int(768 * height / width)
            else:
                new_height = 768
                new_width = int(768 * width / height)

            self.image_window = Gtk.Window(title=embedding_file_name)
            self.image_window.set_default_size(new_width, new_height)

            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(image_path, new_width, new_height)
            img_widget = Gtk.Image.new_from_pixbuf(pixbuf)
            self.image_window.add(img_widget)

            self.image_window.show_all()

    def add_to_positive_prompt_button_pressed(self, widget):
        if self.positive_prompt_field is None or self.selected_embedding_file_name is None:
            return
        
        text_to_add = f"<embedding:{self.selected_embedding_file_name}>"
        positive_prompt = text_view_get_text(self.positive_prompt_field)
        if text_to_add not in positive_prompt:
            text_view_set_text(self.positive_prompt_field, text_to_add + ", " + positive_prompt)

    def add_to_negative_prompt_button_pressed(self, widget):
        if self.negative_prompt_field is None or self.selected_embedding_file_name is None:
            return
        
        text_to_add = f"<embedding:{self.selected_embedding_file_name}>"
        negative_prompt = text_view_get_text(self.negative_prompt_field)
        if text_to_add not in negative_prompt:
            text_view_set_text(self.negative_prompt_field, text_to_add + ", " + negative_prompt)

    def analyze_pressed(self, widget):
        print("Analyze pressed")
        if self.selected_embedding_file_name:
            embedding_file_name = self.selected_embedding_file_name
            embedding_file_full_path = os.path.join(self.embedding_dir_path, embedding_file_name)
            embedding = load_embedding(embedding_file_full_path)
            if isinstance(embedding, dict) is False:  # SD 1.5
                shape_data = str(embedding.shape)
            else: # SDXL
                print(embedding)
                if "clip_g" in embedding and "clip_l" in embedding:
                    shape_data = f'clip_g: {embedding["clip_g"].shape}\nclip_l: {embedding["clip_l"].shape}'
                    # clip_g: torch.Size([52, 1280])
                    # clip_l: torch.Size([52, 768])
                else:
                    show_error_dialog("Unsupported embedding type")

            text_view_set_text(self.embedding_shape_text, shape_data)

# For testing
if __name__ == "__main__":
    css_provider = Gtk.CssProvider()
    css_provider.load_from_data(b"""
        .selected {
            border: 2px solid blue;
            background-color: lightgray;
        }
    """)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win = EmbeddingFileViewerWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
