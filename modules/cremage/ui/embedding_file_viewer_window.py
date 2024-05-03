
import os
import sys
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_set_text, create_combo_box
from cremage.utils.ml_utils import load_torch_model_paths, load_embedding


SELECTION_MESSAGE = "Select the embedding file"

class EmbeddingFileViewerWindow(Gtk.Window):

    def __init__(self, embedding_path=None):
        super().__init__(title="Embedding File Viewer")
        self.embedding_dir_path = embedding_path
        self.set_border_width(10)

        vboxContainer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add(vboxContainer)

        rootScrolledWindow = Gtk.ScrolledWindow()
        rootScrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC) # Show scrollbars as necessary

        # Here we set the minimum size of the scrolled window's content
        rootScrolledWindow.set_min_content_width(150)
        rootScrolledWindow.set_min_content_height(100)

        # Add the ScrolledWindow to vboxContainer instead of adding vboxRoot directly
        vboxContainer.pack_start(rootScrolledWindow, True, True, 0)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
        self.set_default_size(350, 100)

        # Drop down list for embedding files
        self.embedding_file_paths = load_torch_model_paths(embedding_path)
        self.embedding_file_combo_box = create_combo_box([SELECTION_MESSAGE] + self.embedding_file_paths)
        
        vbox.pack_start(self.embedding_file_combo_box,
                        False,  # Do not expand even if the parent window expands
                        True, # Fit the allocated space initially
                        0)

        # Embedding shape button
        self.embedding_shape_text = Gtk.TextView()
        self.embedding_shape_text.set_wrap_mode(Gtk.WrapMode.WORD)
        vbox.pack_start(self.embedding_shape_text, False, True, 0)

        # Analyze button
        self.analyze_button = Gtk.Button(label="Analyze")
        self.analyze_button.connect("clicked", self.analyze_pressed)
        vbox.pack_start(self.analyze_button, False, True, 0)

        # Add vbox to the ScrolledWindow
        rootScrolledWindow.add_with_viewport(vbox)


    def analyze_pressed(self, widget):
        print("Analyze pressed")
        embedding_file_name = self.embedding_file_combo_box.get_active_text()
        if embedding_file_name == SELECTION_MESSAGE:
            return
        else:
            embedding_file_full_path = os.path.join(self.embedding_dir_path, embedding_file_name)
            embedding = load_embedding(embedding_file_full_path)
            text_view_set_text(self.embedding_shape_text, str(embedding.shape))
    


