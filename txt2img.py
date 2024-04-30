"""
Text to image generator main script.
"""
import image_generator
from options import parse_options

def generate(opt,
             ui_thread_instance=None,
             status_queue=None):
    
    image_generator.generate(opt,
                             ui_thread_instance=ui_thread_instance,
                             generation_type="txt2img",
                             status_queue=status_queue)


def parse_options_and_generate(args=None,
                               ui_thread_instance=None,
                               status_queue=None
                               ):
    opt = parse_options(args)    
    generate(opt, ui_thread_instance=ui_thread_instance, status_queue=status_queue)


if __name__ == "__main__":
    parse_options_and_generate()
