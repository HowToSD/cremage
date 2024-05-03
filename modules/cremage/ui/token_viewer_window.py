import os
import sys
from transformers import CLIPTextModel
from transformers import CLIPTokenizerFast as CLIPTokenizer
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_get_text

class TokenListItem(Gtk.ListBoxRow):
    def __init__(self, index, token, word):
        super().__init__()
        self.box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.add(self.box)

        # Create the index label with a fixed width of 60 pixels
        self.index_label = Gtk.Label(label=str(index))
        self.index_label.set_size_request(60, -1)  # Set fixed width to 60px
        self.box.pack_start(self.index_label, False, False, 0)

        # Create the token label with a fixed width of 100 pixels
        self.token_label = Gtk.Label(label=str(token))
        self.token_label.set_size_request(100, -1)  # Set fixed width to 100px
        self.box.pack_start(self.token_label, False, False, 0)

        # Create the word label without a fixed width
        self.word_label = Gtk.Label(label=word)
        self.box.pack_start(self.word_label, True, True, 0)


class TokenViewerWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Token Viewer")
        self.set_border_width(10)

        vboxContainer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add(vboxContainer)

        rootScrolledWindow = Gtk.ScrolledWindow()
        rootScrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC) # Show scrollbars as necessary

        # Here we set the minimum size of the scrolled window's content
        rootScrolledWindow.set_min_content_width(1200)
        rootScrolledWindow.set_min_content_height(1000)

        # Add the ScrolledWindow to vboxContainer instead of adding vboxRoot directly
        vboxContainer.pack_start(rootScrolledWindow, True, True, 0)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
        self.set_default_size(400, 600)

        # Input Prompt
        self.input_prompt = Gtk.TextView()
        vbox.pack_start(self.input_prompt, True, True, 0)

        # Tokens gridbox
        self.tokens_gridbox = Gtk.ListBox()
        vbox.pack_start(self.tokens_gridbox, True, True, 0)

        # Tokenize button
        self.tokenize_button = Gtk.Button(label="Tokenize")
        self.tokenize_button.connect("clicked", self.tokenized_pressed)
        vbox.pack_start(self.tokenize_button, False, True, 0)

        # Add vbox to the ScrolledWindow
        rootScrolledWindow.add_with_viewport(vbox)


    def tokenized_pressed(self, widget):
        text = text_view_get_text(self.input_prompt)
        
        # Clear Tokens gridbox
        for child in self.tokens_gridbox.get_children():
            self.tokens_gridbox.remove(child)

        version = "openai/clip-vit-large-patch14"  # Example version
        device = os.environ.get("GPU_DEVICE", "cpu")
        max_length = 77  # Set max_length as per CLIP's requirement

        # Initialize tokenizer and model
        tokenizer = CLIPTokenizer.from_pretrained(version)
        transformer = CLIPTextModel.from_pretrained(version)

        # Encode the input text
        batch_encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True  # Important for mapping
        )
        tokens = batch_encoding["input_ids"].to(device)
        offset_mapping = batch_encoding["offset_mapping"][0]  # Get the offsets for the first (and only) encoded input

        # offset is a pair of int which contains start and end of the word in the text array.
        # Initialize an empty list to store mappings
        word_token_mappings = []

        # Iterate through tokens and their offset mappings
        for token_id, offsets in zip(tokens[0], offset_mapping):
            # Extract the start and end offsets
            start, end = offsets
            if start == end:  # Special tokens have (0, 0) as offsets
               ## word = "special"
                word = tokenizer.decode([token_id])
            else:
                # Extract the word from the original text using offsets
                word = text[start:end]

            # Append the word and its corresponding token ID to the list
            word_token_mappings.append((word, token_id.item()))

        # Print the mappings
        generated_tokens = list()
        for i, (word, token_id) in enumerate(word_token_mappings):
            print(f"{word}, {token_id}")
            generated_tokens.append({
                "index": i, "token": token_id, "word": word
            })

        for token in generated_tokens:
            item = TokenListItem(token["index"], token["token"], token["word"])
            self.tokens_gridbox.add(item)
        self.tokens_gridbox.show_all()

