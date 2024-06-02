import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.generate_open_clip_embeddings_from_tokens import token_to_embedding
from cremage.utils.generate_open_clip_embeddings_from_tokens import convert_word_to_tokens
from cremage.utils.generate_open_clip_embeddings_from_tokens import generate_open_clip_embeddings
from cremage.utils.prompt_score_parser import _compute_product_score
from cremage.utils.prompt_score_parser import _build_tree


class TestPromptScoreParserComputeProductScore(unittest.TestCase):


    def test_score_1(self):
        text = "hello"
        root = _build_tree(text)
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)

    def test_score_2(self):
        text = "hello(abc)x"
        root = _build_tree(text)
        child = root.children[0]
        child.score = 1.1
        _compute_product_score(root, root.score)
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        self.assertTrue(child.score==1.1)
        self.assertTrue(child.product_score==1.1)

    def test_score_3(self):
        text = "hello(abc)x(def)"
        root = _build_tree(text)
        child1 = root.children[0]
        child2 = root.children[1]
        child1.score = 1.1
        child2.score = 1.1
        _compute_product_score(root, root.score)
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        self.assertTrue(child1.score==1.1)
        self.assertTrue(child1.product_score==1.1)
        self.assertTrue(child2.score==1.1)
        self.assertTrue(child2.product_score==1.1)

    def test_score_4(self):
        text = "(hello)"
        root = _build_tree(text)
        child = root.children[0]
        child.score = 1.1
        _compute_product_score(root, root.score)
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        self.assertTrue(child.score==1.1)
        self.assertTrue(child.product_score==1.1)

    def test_score_5(self):
        text = "hello(abc(def))x"
        root = _build_tree(text)
        child1 = root.children[0]
        child2 = child1.children[0]
        child1.score = 1.1
        child2.score = 1.1
        _compute_product_score(root, root.score)
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        self.assertTrue(child1.score==1.1)
        self.assertTrue(child1.product_score==1.1)
        self.assertTrue(child2.score==1.1)
        self.assertTrue(round(child2.product_score, 2)==1.21)


if __name__ == '__main__':
    unittest.main()
