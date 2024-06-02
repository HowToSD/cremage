import os
import sys
import json
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, Pango

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.const.const import *
from cremage.utils.gtk_utils import text_view_set_text
from cremage.utils.prompt_history import remove_prompt_entry
from cremage.utils.prompt_history import prompt_data_for_data_path

class PromptHistoryViewer(Gtk.Window):

    def __init__(self, data_path, update_handler=None):
        Gtk.Window.__init__(self, title="Prompt History")
        self.set_default_size(400, 300)
        self.update_handler = update_handler
        self.data_path = data_path
        self.prompts_data = prompt_data_for_data_path(self.data_path)

        self.vbox = Gtk.VBox(spacing=10)
        self.vbox.set_margin_top(10)
        self.vbox.set_margin_bottom(10)
        self.vbox.set_margin_left(10)
        self.vbox.set_margin_right(10)
        self.add(self.vbox)

        # Add search label and entry
        self.search_box = Gtk.Box(spacing=5)
        self.search_label = Gtk.Label(label="Search")
        self.search_entry = Gtk.Entry()
        self.search_entry.connect("changed", self.on_search_entry_changed)
        self.search_box.pack_start(self.search_label, False, False, 0)
        self.search_box.pack_start(self.search_entry, True, True, 0)
        self.vbox.pack_start(self.search_box, False, False, 0)
        
        # Create a scrolled window and add the listbox to it
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.listbox = Gtk.ListBox()
        self.scrolled_window.add(self.listbox)
        
        # Add the scrolled window to the vertical box
        self.vbox.pack_start(self.scrolled_window, True, True, 0)
        
        self.button_box = Gtk.Box(spacing=10)
        self.vbox.pack_start(self.button_box, False, False, 0)
        
        self.copy_button = Gtk.Button(label="Copy")
        self.copy_button.connect("clicked", self.on_copy_button_clicked)
        self.button_box.pack_start(self.copy_button, True, True, 0)
        
        self.close_button = Gtk.Button(label="Close")
        self.close_button.connect("clicked", self.on_close_button_clicked)
        self.button_box.pack_start(self.close_button, True, True, 0)
        
        self.populate_listbox()

        # Connect key-press-event to handle delete key
        self.connect("key-press-event", self.on_key_press)
        # self.prompts_data = None

    def populate_listbox(self, filter_text=""):
        # Clear existing listbox entries
        for row in self.listbox.get_children():
            self.listbox.remove(row)
        
        prompts = reversed(self.prompts_data["prompts"])
        for row in prompts:
            if filter_text.lower() in row.lower():
                listbox_row = Gtk.ListBoxRow()
                
                # Create a frame to add border
                frame = Gtk.Frame()
                frame.set_shadow_type(Gtk.ShadowType.IN)
                
                label = Gtk.Label(label=row)
                label.set_xalign(0)  # Left align the text
                label.set_line_wrap(True)  # Enable word wrap
                label.set_line_wrap_mode(Pango.WrapMode.WORD)  # Set wrap mode to word
                label.set_max_width_chars(50)  # Set maximum width for wrapping
                
                frame.add(label)
                listbox_row.add(frame)
                self.listbox.add(listbox_row)

        self.listbox.show_all()

    def on_search_entry_changed(self, widget):
        search_text = widget.get_text()

        print("debug: " + search_text)

        if search_text == "":
            self.populate_listbox()  # Show all prompts when the search text is empty
        else:
            self.populate_listbox(search_text)

    def on_copy_button_clicked(self, widget):
        selected_row = self.listbox.get_selected_row()
        if selected_row:
            frame = selected_row.get_child()
            label = frame.get_child()
            prompt = label.get_text()
            if self.update_handler:
                self.update_handler(prompt)
        else:
            self.show_error_dialog("No row is selected")

    def on_close_button_clicked(self, widget):
        self.destroy()

    def show_error_dialog(self, message):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def on_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_Delete:
            selected_row = self.listbox.get_selected_row()
            if selected_row:
                self.delete_entry(selected_row)

    def delete_entry(self, row):
        frame = row.get_child()
        label = frame.get_child()
        prompt = label.get_text()
        remove_prompt_entry(self.data_path, prompt)
        
        # Remove from Gtk.ListBox
        self.listbox.remove(row)


if __name__ == "__main__":
    import os

    json_path = os.path.expanduser("~/.cremage/data/history/positive_prompts.json")
    win = PromptHistoryViewer(json_path)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
