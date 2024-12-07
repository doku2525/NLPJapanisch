from unittest import TestCase
import copy

from src.parser.NLPMecabParser import MecabParser


class test_MecabParser(TestCase):

    def setUp(self):
        self.obj_a = MecabParser.parseText("お煎茶を美味しく飲んだ。")
        self.obj_b = MecabParser.parseText("人々のこのような意気込みにもかかわらず、３ヶ月間で生産された食糧はわずか２７３トンと、年間目標の３５００トンには遠く及びませんでした。")
        self.obj_c = MecabParser()
        self.obj = copy.copy(self.obj_a)

    def test_init(self):
        self.assertEqual("", self.obj_c.source)
        self.assertIsNone(self.obj_c.mecabstring)

    def test_parse_text(self):
        for obj in [self.obj_a, self.obj_b]:
            self.assertIs(type(obj.mecabstring), str)
            self.assertGreater(len(obj.mecabstring), len(obj.source))

    def test_as_matrix(self):
        from typing import cast

        for obj in [self.obj_a, self.obj_b]:
            matrix = obj.as_matrix()
            self.assertIs(type(matrix), list)
            self.assertGreater(len(matrix), 0)
            self.assertLess(len(matrix), 43)
            self.assertEqual(len(matrix[0]), 30)
            self.assertEqual(len(matrix[-1]), 30)
            self.assertEqual("".join(cast(list[str], matrix[-1])), "EOS"*30)
            self.assertListEqual([len(x) for x in matrix[:-2]],
                                 [30 for x in matrix[:-2]])

    def test_tokenizer_for_count_vectorizer(self):
        result = self.obj.tokenizer_for_count_vectorizer(self.obj.source)
        matrix = self.obj.as_matrix()
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(matrix) - 1)  # result ist ohne 'EOS'
        self.assertEqual("".join(result), self.obj.source)

    def test_column_from_matrix(self):
        result = self.obj.column_from_matrix(0)
        self.assertIs(type(result), list)
        self.assertIs(type(result[0]), str)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], 'お')
        self.assertEqual(result[1], '煎茶')
        self.assertNotEquals(result[-1], 'EOS')
