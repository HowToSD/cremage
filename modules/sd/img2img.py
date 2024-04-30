"""
Image to image generation script
"""
from sd import image_generator
from sd.options import parse_options

def generate(opt,
             ui_thread_instance=None,
             status_queue=None):
    
    image_generator.generate(opt,
                             ui_thread_instance=ui_thread_instance,
                             generation_type="img2img",
                             status_queue=status_queue)
    
def img2img_parse_options_and_generate(args=None,
                                       ui_thread_instance=None,
                                       status_queue=None):
    opt = parse_options(args)    
    generate(opt, ui_thread_instance=ui_thread_instance, status_queue=status_queue)


if __name__ == "__main__":
    img2img_parse_options_and_generate()
