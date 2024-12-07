from unittest import TestCase
import copy

from src.parser.NLPMecabParser import MecabParser, MecabPOS


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
            self.assertIsInstance(obj.mecabstring, str)
            self.assertGreater(len(obj.mecabstring), len(obj.source))

    def test_as_matrix(self):
        from typing import cast

        for obj in [self.obj_a, self.obj_b]:
            matrix = obj.as_matrix()
            self.assertIsInstance(matrix, list)
            self.assertGreater(len(matrix), 0)
            self.assertLess(len(matrix), 43)
            self.assertEqual(len(matrix[0]), 30)
            self.assertEqual(len(matrix[-1]), 30)
            self.assertEqual("".join(cast(list[str], matrix[-1])), "EOS"*30)
            self.assertListEqual([len(x) for x in matrix[:-2]],
                                 [30 for x in matrix[:-2]])

    def test_as_wordlist_for_count_vectorizer(self):
        result = self.obj.as_wordlist_for_count_vectorizer()
        matrix = self.obj.as_matrix()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(matrix) - 1)  # result ist ohne 'EOS'
        self.assertEqual("".join(result), self.obj.source)

    def test_tokenizer_for_count_vectorizer(self):
        result = MecabParser.tokenizer_for_count_vectorizer(self.obj.source)
        matrix = self.obj.as_matrix()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(matrix) - 1)  # result ist ohne 'EOS'
        self.assertEqual("".join(result), self.obj.source)

    def test_get_position_from_matrix(self):
        result = self.obj.get_position_from_matrix(0)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], str)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], 'お')
        self.assertEqual(result[1], '煎茶')
        self.assertNotEquals(result[-1], 'EOS')

    def test_mecabpos(self):
        self.assertEqual(self.obj.get_position_from_matrix(0),
                         self.obj.get_position_from_matrix(MecabPOS.WORT.value))

    def test_get_positionwindow_from_matrix(self):
        result = self.obj.get_positionwindow_from_matrix(slice(0, 2))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertEqual(2, len(result[0]))
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0][0], 'お')
        self.assertEqual(result[0][1], '接頭辞')
        self.assertEqual(result[1][0], '煎茶')
        self.assertEqual(result[1][1], '名詞')
        self.assertNotEquals(result[-1][0], 'EOS')
        self.assertNotEquals(result[-1][1], 'EOS')
        result = self.obj.get_positionwindow_from_matrix(slice(0, 200))
        self.assertEqual(self.obj.normale_token_laenge, len(result[0]))
        result = self.obj.get_positionwindow_from_matrix(slice(0, self.obj.normale_token_laenge))
        self.assertEqual(self.obj.normale_token_laenge, len(result[0]))
