import os
import sys
import unittest

from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TEST_IMAGE_FILE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "real1_blurred.jpg")
sys.path = [MODULE_ROOT] + sys.path

from unblur_face.face_unblur import unblur_face_image

class TestUnblurFace(unittest.TestCase):
    def test_unblur(self):
        pil_image = Image.open(TEST_IMAGE_FILE_PATH)
        out_image = unblur_face_image(pil_image)
        out_image.save("/var/tmp/unblur_test.jpg")
        self.assertTrue(out_image.size == (768, 512))


if __name__ == '__main__':
    unittest.main()