"""
Defines the event handler for generator type (SD 1.5 vs SDXL) change.
"""
import os
import logging
import sys
import subprocess
import platform

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path
from cremage.const.const import *
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.utils.app_misc_utils import get_next_face_file_path
from cremage.utils.misc_utils import open_os_directory

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def generator_model_type_changed(app, combo):
    generator_model_type = combo.get_child().get_text()
    toggle_genenator_model_type_ui(app, generator_model_type)


def toggle_genenator_model_type_ui(app:Gtk.Window, generator_model_type):

    if generator_model_type == GMT_SD_1_5 or \
       generator_model_type in [
           GMT_SD_3,
           GMT_STABLE_CASCADE,
           GMT_PIXART_SIGMA,
           GMT_KANDINSKY_2_2,
           GMT_HUNYUAN_DIT]:

        # Common between SD 1.5 and non-SDXL
        app.fields["sdxl_sampler"].hide()
        app.fields["image_width"].show()
        app.fields["image_height"].show()
        app.fields["sdxl_image_resolution"].hide()

        app.fields1_labels["Sdxl sampler"].hide()
        app.fields1_labels["Image width"].show()
        app.fields1_labels["Image height"].show()
        app.fields1_labels["Sdxl image resolution"].hide()

        sdxl_page_num = app.notebook.page_num(app.tab_contents["SDXL Models"])
        if sdxl_page_num >= 0:
            app.notebook.remove_page(sdxl_page_num)

        sdxl_advanced_page_num = app.notebook.page_num(app.tab_contents["SDXL Advanced"])
        if sdxl_advanced_page_num >= 0:
            app.notebook.remove_page(sdxl_advanced_page_num)

        if generator_model_type == GMT_SD_1_5:
            pixart_sigma_page_num = app.notebook.page_num(app.tab_contents["PixArt-Sigma"])
            if pixart_sigma_page_num >= 0:
                app.notebook.remove_page(pixart_sigma_page_num)

            app.fields["sampler"].show()
            app.fields1_labels["Sampler"].show()
            app.fields["clip_skip"].show()
            app.fields1_labels["Clip skip"].show()
            app.rb_image_to_image.show()
            app.rb_inpainting.show()

            app.fields["hires_fix_upscaler"].show()
            app.fields1_labels["Hires fix upscaler"].show()
            app.fields["hires_fix_scale_factor"].show()
            app.fields1_labels["Hires fix scale factor"].show()
            app.fields["denoising_strength"].show()
            app.fields1_labels["Denoising strength"].show()

            sd15_page_num = app.notebook.page_num(app.tab_contents["Models"])
            if sd15_page_num == -1:
                app.notebook.insert_page(
                    app.tab_contents["Models"],
                    app.tab_labels["Models"],
                    1)

            control_net_page_num = app.notebook.page_num(app.tab_contents["ControlNet"])
            if control_net_page_num == -1:
                app.notebook.insert_page(
                    app.tab_contents["ControlNet"],
                    app.tab_labels["ControlNet"],
                    2)

        elif generator_model_type in \
            [
                GMT_SD_3,
                GMT_STABLE_CASCADE,
                GMT_PIXART_SIGMA,
                GMT_KANDINSKY_2_2,
                GMT_HUNYUAN_DIT
            ]:
            app.fields["sampler"].hide()
            app.fields1_labels["Sampler"].hide()
            app.fields["clip_skip"].hide()
            app.fields1_labels["Clip skip"].hide()

            app.fields["hires_fix_upscaler"].hide()
            app.fields1_labels["Hires fix upscaler"].hide()
            app.fields["hires_fix_scale_factor"].hide()
            app.fields1_labels["Hires fix scale factor"].hide()

            if generator_model_type in [GMT_SD_3, GMT_STABLE_CASCADE, GMT_HUNYUAN_DIT]:
                pixart_sigma_page_num = app.notebook.page_num(app.tab_contents["PixArt-Sigma"])
                if pixart_sigma_page_num >= 0:
                    app.notebook.remove_page(pixart_sigma_page_num)

                app.fields["denoising_strength"].hide()
                app.fields1_labels["Denoising strength"].hide()
                app.rb_image_to_image.hide()
                app.rb_inpainting.hide()

            elif generator_model_type in [GMT_PIXART_SIGMA]:

                pixart_sigma_page_num = app.notebook.page_num(app.tab_contents["PixArt-Sigma"])
                if pixart_sigma_page_num  == -1:
                    app.notebook.insert_page(
                        app.tab_contents["PixArt-Sigma"],
                        app.tab_labels["PixArt-Sigma"],
                        1)

                app.fields["denoising_strength"].hide()
                app.fields1_labels["Denoising strength"].hide()
                app.rb_image_to_image.hide()
                app.rb_inpainting.hide()

                app.fields["denoising_strength"].hide()
                app.fields1_labels["Denoising strength"].hide()
                app.rb_image_to_image.hide()
                app.rb_inpainting.hide()

            elif generator_model_type in [GMT_KANDINSKY_2_2]:
                pixart_sigma_page_num = app.notebook.page_num(app.tab_contents["PixArt-Sigma"])
                if pixart_sigma_page_num >= 0:
                    app.notebook.remove_page(pixart_sigma_page_num)

                app.fields["denoising_strength"].show()
                app.fields1_labels["Denoising strength"].show()
                app.rb_image_to_image.show()
                app.rb_inpainting.show()

            sd15_page_num = app.notebook.page_num(app.tab_contents["Models"])
            if sd15_page_num >= 0:
                app.notebook.remove_page(sd15_page_num)

            control_net_page_num = app.notebook.page_num(app.tab_contents["ControlNet"])
            if control_net_page_num >= 0:
                app.notebook.remove_page(control_net_page_num)

    elif generator_model_type == "SDXL":
        app.fields["sampler"].hide()
        app.fields["sdxl_sampler"].show()
        app.fields["image_width"].hide()
        app.fields["image_height"].hide()
        app.fields["sdxl_image_resolution"].show()
        app.fields["clip_skip"].hide()

        app.fields1_labels["Sampler"].hide()
        app.fields1_labels["Sdxl sampler"].show()
        app.fields1_labels["Image width"].hide()
        app.fields1_labels["Image height"].hide()
        app.fields1_labels["Sdxl image resolution"].show()
        app.fields1_labels["Clip skip"].hide()

        app.fields["hires_fix_upscaler"].show()
        app.fields1_labels["Hires fix upscaler"].show()
        app.fields["hires_fix_scale_factor"].show()
        app.fields1_labels["Hires fix scale factor"].show()
        app.fields["denoising_strength"].show()
        app.fields1_labels["Denoising strength"].show()

        app.rb_image_to_image.show()
        app.rb_inpainting.hide()
        
        app.tab_labels["Models"].set_visible(False)
        app.tab_labels["ControlNet"].set_visible(False)
        app.tab_labels["SDXL Models"].set_visible(True)
        app.tab_labels["SDXL Advanced"].set_visible(True)
        sd15_page_num = app.notebook.page_num(app.tab_contents["Models"])
        if sd15_page_num >= 0:
            app.notebook.remove_page(sd15_page_num)

        control_net_page_num = app.notebook.page_num(app.tab_contents["ControlNet"])
        if control_net_page_num >= 0:
            app.notebook.remove_page(control_net_page_num)

        pixart_sigma_page_num = app.notebook.page_num(app.tab_contents["PixArt-Sigma"])
        if pixart_sigma_page_num >= 0:
            app.notebook.remove_page(pixart_sigma_page_num)

        sdxl_page_num = app.notebook.page_num(app.tab_contents["SDXL Models"])
        if sdxl_page_num == -1:
            app.notebook.insert_page(
                app.tab_contents["SDXL Models"],
                app.tab_labels["SDXL Models"],
                1)

        sdxl_advanced_page_num = app.notebook.page_num(app.tab_contents["SDXL Advanced"])
        if sdxl_advanced_page_num == -1:
            app.notebook.insert_page(
                app.tab_contents["SDXL Advanced"],
                app.tab_labels["SDXL Advanced"],
                5)
