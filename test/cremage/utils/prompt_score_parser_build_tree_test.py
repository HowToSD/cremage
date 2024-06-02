import os
import sys
import unittest


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path

from cremage.utils.prompt_score_parser import _build_tree


class TestPromptScoreParserBuildTree(unittest.TestCase):

    def test_build_tree_1(self):
        text = "hello"
        root = _build_tree(text)
        self.assertTrue(root.characters==["h", "e", "l", "l", "o"])
        self.assertTrue(root.pos==[0, 1, 2, 3, 4])
        
    def test_build_tree_2(self):
        text = "hello(abc)x"
        root = _build_tree(text)
        self.assertTrue(root.characters==["h", "e", "l", "l", "o", "x"])
        self.assertTrue(root.pos==[0, 1, 2, 3, 4, 10])
        child = root.children[0]
        self.assertTrue(child.characters==["a", "b", "c"])
        self.assertTrue(child.pos==[6, 7, 8])  # ( is counted in index so 6 instead of 5.

    def test_build_tree_3(self):
        text = "(hello)"
        root = _build_tree(text)
        self.assertTrue(root.characters==[])
        self.assertTrue(root.pos==[])
        child = root.children[0]
        self.assertTrue(child.characters==["h", "e", "l", "l", "o"])
        self.assertTrue(child.pos==[1, 2, 3, 4, 5])
        
    def test_build_tree_4(self):
        text = "hello(abc)x(def)"
        root = _build_tree(text)
        self.assertTrue(root.characters==["h", "e", "l", "l", "o", "x"])
        self.assertTrue(root.pos==[0, 1, 2, 3, 4, 10])
        child = root.children[0]
        self.assertTrue(child.characters==["a", "b", "c"])
        self.assertTrue(child.pos==[6, 7, 8])  # ( is counted in index so 6 instead of 5.
        child = root.children[1]
        self.assertTrue(child.characters==["d", "e", "f"])
        self.assertTrue(child.pos==[12, 13, 14])

    def test_build_tree_5(self):
        text = "hello(abc(def))x"
        root = _build_tree(text)
        self.assertTrue(root.characters==["h", "e", "l", "l", "o", "x"])
        self.assertTrue(root.pos==[0, 1, 2, 3, 4, 15])
        child = root.children[0]
        self.assertTrue(child.characters==["a", "b", "c"])
        self.assertTrue(child.pos==[6, 7, 8])
        child = child.children[0]
        self.assertTrue(child.characters==["d", "e", "f"])
        self.assertTrue(child.pos==[10, 11, 12])


if __name__ == '__main__':
    unittest.main()
