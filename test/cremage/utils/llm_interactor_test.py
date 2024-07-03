import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [PROJECT_ROOT] + sys.path
from tools.llm_interactor import extract_response

class TestLLMInteractor(unittest.TestCase):

    def test_inst_extraction(self):
        """
        Check if response extraction is working.
        """
        expected_response = "That's great! Dogs are wonderful companions. What kind of dog is Jean?"
        raw_response = """
[INST] I'm Tom. [/INST] Hello Tom! How can I help you today?

[INST] I have a dog named Jean.  [/INST] """ + expected_response

        response = extract_response(raw_response)
        self.assertTrue(response == expected_response, f"Error. Wrong result: {response}")


if __name__ == '__main__':
    unittest.main()
