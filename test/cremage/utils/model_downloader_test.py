import os
import sys
import unittest


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "unblur_face")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.model_downloader import download_model_if_not_exist

class TestModelDownloader(unittest.TestCase):


    def test_model_downloader(self):
        """
        Check if model downloader is working as expected.
        """
        tmp_dir = "/var/tmp"
        model_name = "face_unblur_v11_epoch51.pth"
        model_path = os.path.join(tmp_dir, model_name)
        self.assertFalse(os.path.exists(model_path))
        model_path = download_model_if_not_exist(
            tmp_dir,
            "HowToSD/face_unblur",
            model_name,
            )
        self.assertTrue(os.path.exists(model_path))
        if model_path.startswith(tmp_dir):
            os.remove(model_path)
            self.assertFalse(os.path.exists(model_path))
        else:
            self.assertTrue(False)  # invalid model path

if __name__ == '__main__':
    unittest.main()
