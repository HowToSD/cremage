"""
Image to image generation script
"""
from sd import image_generator
# from sd.options import parse_options

def generate(options=None,
             ui_thread_instance=None,
             status_queue=None):
    
    image_generator.generate(options,
                             ui_thread_instance=ui_thread_instance,
                             generation_type="img2img",
                             status_queue=status_queue)
