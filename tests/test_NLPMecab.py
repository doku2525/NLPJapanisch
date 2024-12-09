from unittest import TestCase

from src.NLPMecab import NLPMecab
from src.parser.NLPMecabParser import MecabParser


class test_NLPJapanischMecab(TestCase):

    def setUp(self):
        self.obj = NLPMecab()
        self.sentences = ["これには私ばかりでなく、ほかの多くの人も不思議に思った。",
                          "事実はたんの小説よりも、奇なるばかりではなく、より劇的だよ。",
                          "彼は、この女の人が美しいばかりでなく、頭もよさそうなことを見てとった。",
                          "私は土地を失ったばかりでなく、それにつぎ込んだ心血まで、すっかりなくしてしまった。",
                          "たぶん天皇は、民衆の苦しみの象徴であるばかりでなく、この負け戦の最大の犠牲者であったのだろう。",
                          "彼女は、たいていの人がそうしたいという気を起こすように、事実を曲げようともしなかったばかりか、驚くほどの率直さで、まっすぐ要点をいってのけたのだ。",
                          "友人は自身演奏がたいへん巧みであるばかりでなく、作曲にかけてもなみなみならぬ手腕をもっているという、音楽熱心の男である。",
                          "連合軍の注意を主攻撃からそらしたばかりでなく、連合軍機動兵力の大部分を同方面に吸収してしまった。",
                          "ピーターさんは日本に行くことは、友人ばかりでなく、家族でさえも知りませんでした。"
                          ]
        self.matrix = NLPMecab.build_matrix_from_list_of_sentences(MecabParser.parseText, self.sentences)

    def test_build_matrix_from_list(self):
        self.assertIsInstance(self.matrix, list)
        self.assertIsInstance(self.matrix[0], list)
        self.assertIsInstance(self.matrix[0][0], list)
        self.assertIsInstance(self.matrix[0][0][0], str)
        self.assertEqual(len(self.matrix), len(self.sentences))
        self.assertEqual(self.matrix[0][0][0], "これ")
        self.assertEqual(self.matrix[-1][0][0], "ピーター")
        [self.assertEqual(len(token), 30, token) for saetze in self.matrix for token in saetze]

    def test_create_from_matrix(self):
        self.assertEqual(self.obj.max_y, 0)
        self.assertEqual(self.obj.max_x, 0)
        self.assertEqual(self.obj.matrix, [])
        self.obj.create_from_matrix(self.matrix)
        self.assertEqual(self.obj.max_y, len(self.sentences))
        self.assertEqual(self.obj.max_y, len(self.obj.matrix))
        self.assertEqual(self.obj.max_x, 47)

    def test_number_of_tokens_per_sentence(self):
        self.assertEquals(max(self.obj.number_of_tokens_per_sentence()), self.obj.max_x)
        self.assertEquals(max(self.obj.number_of_tokens_per_sentence()), 0)
        self.obj.create_from_matrix(self.matrix)
        self.assertIs(type(self.obj.number_of_tokens_per_sentence()), list)
        self.assertEqual(len(self.obj.number_of_tokens_per_sentence()), len(self.matrix))
        self.assertEqual(len(self.obj.number_of_tokens_per_sentence()), self.obj.max_y)
        self.assertGreater(min(self.obj.number_of_tokens_per_sentence()), 1)
        self.assertEquals(max(self.obj.number_of_tokens_per_sentence()), self.obj.max_x)

    def test_get_max_x(self):
        self.assertEqual(self.obj.get_max_x(), 0)
        self.assertEqual(self.obj.max_number_of_tokens_per_sentence(), 0)
        self.obj.create_from_matrix(self.matrix)
        self.assertEqual(self.obj.get_max_x(), 47)
        self.assertEqual(self.obj.max_number_of_tokens_per_sentence(), 47)

    def test_get_mecab_info_at_position_per_sentence(self):
        self.obj.create_from_matrix(self.matrix)
        self.assertEqual(len(self.obj.get_mecab_info_at_position_per_sentence(0)), self.obj.number_of_sentences)
        self.assertEqual(len(self.obj.get_mecab_info_at_position_per_sentence(0)[0]), self.obj.tokens_per_sentence[0])
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(0)[0][0], 'これ')
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(0)[0][1], 'に')
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(0)[1][0], '事実')
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(0)[2][0], '彼')
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(1)[0][0], '代名詞')
        self.assertTrue(self.obj.get_mecab_info_at_position_per_sentence(29)[0][0].isdigit())
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(29)[-1][-1], 'EOS')
        self.assertEqual(self.obj.get_mecab_info_at_position_per_sentence(20)[-1][-1], 'EOS')

    def test_trim_sentences_to_length(self):
        self.obj.create_from_matrix(self.matrix)
        for sentence in self.obj.matrix:
            self.assertEqual(len(self.obj.trim_sentence_to_length(sentence, 60)), 60)
            self.assertEqual(len(self.obj.trim_sentence_to_length(sentence, 10)), 10)
            for token in self.obj.trim_sentence_to_length(sentence, 60):
                self.assertEqual(len(token), 30, token)
            for token in self.obj.trim_sentence_to_length(sentence, 10):
                self.assertEqual(len(token), 30, token)

    def test_as_numpy_array(self):
        from numpy import ndarray

        self.obj.create_from_matrix(self.matrix)
        self.assertIsInstance(self.obj.as_numpy_array(), ndarray)
        dimension = (self.obj.number_of_sentences, self.obj.max_x, self.obj.infos_per_token)
        self.assertEqual(self.obj.as_numpy_array().shape, dimension)

    def test_get_vocabulary(self):
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        result = self.obj.get_vocabulary()
        self.assertIs(type(result), dict)
        self.assertEqual(len(result.keys()), max(result.values())+1, "+1, weil der kleinste Wert 0 ist.")
        self.assertEqual(len(result.values()), len(set(result.values())), "Die Werte in value sind einmalig")
        self.assertIn('これ', result.keys())
        self.assertIn('EOS', result.keys())
        self.obj.execute_count_vectorizer_on_info_pos(1)
        result = self.obj.get_vocabulary()
        self.assertIs(type(result), dict)
        self.assertEqual(len(result.keys()), max(result.values())+1)
        self.assertNotIn('これ', result.keys())
        self.assertIn('EOS', result.keys())

    def test_execute_count_vectorizer_on_info_pos(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from scipy.sparse import csr_matrix

        self.obj.create_from_matrix(self.matrix)
        self.assertIsInstance(self.obj.count_vectorizer, CountVectorizer)
        self.assertFalse(self.obj.count_vectorizer_result)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        self.assertIsInstance(self.obj.count_vectorizer, CountVectorizer)
        self.assertIsInstance(self.obj.count_vectorizer_result, csr_matrix)

    def test_get_number_of_occurence_per_sentence_for_word(self):
        from numpy import ndarray
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        result = self.obj.get_number_of_occurence_per_sentence_for_word('これ')
        result2 = self.obj.get_number_of_occurence_per_sentence_for_word('EOS')
        result3 = self.obj.get_number_of_occurence_per_sentence_for_word('、')
        print(f"\n {result} {result2}")
        self.assertIsInstance(result, ndarray)
        self.assertEqual(len(result), self.obj.max_y)
        self.assertEqual(max(result), 1)
        self.assertEqual(max(result2), 1)
        self.assertEqual(sum(result2), self.obj.max_y)
        self.assertEqual(max(result3), 4)
        self.assertEqual(result2[0], 1)

    def test_create_binary_count_quote(self):
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        result = self.obj.create_binary_count_quote('これ')
        self.assertIs(type(result), dict)
        self.assertIn('binary', result.keys())
        self.assertIn('count', result.keys())
        self.assertGreaterEqual(result['count'], result['binary'])

    def test_create_dict_of_all_word_with_binary_count_quote(self):
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        result = self.obj.create_dict_of_all_word_with_binary_count_quote()
        binary_max = max([stats['binary'] for stats in result.values()])
        binary_min = min([stats['binary'] for stats in result.values()])
        count_max = max([stats['count'] for stats in result.values()])
        count_min = min([stats['count'] for stats in result.values()])
        self.assertEqual(binary_min, count_min)
        self.assertEqual(binary_max, self.obj.max_y)
        self.assertGreater(count_max, self.obj.max_y)

    def test_filter_by_binary_count_quote(self):
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        result = self.obj.filter_by_binary_count_quote(lambda value: value >= 8, 'binary')
        self.assertIsInstance(result, dict)
        self.assertIn('ばかり', result.keys())
        self.assertNotIn('これ', result.keys())

    def test_get_wordposition_absolute_in_sentences(self):
        self.obj.create_from_matrix(self.matrix)
        result = self.obj.get_wordposition_absolute_in_sentences('ばかり', 0)
        expected = [[4], [9], [9], [6], [12], [26], [9], [10], [10]]
        self.assertEqual(len(result), self.obj.max_y)
        self.assertEqual(result, expected)

    def test_get_wordposition_in_percent_in_sentences(self):
        self.obj.create_from_matrix(self.matrix)
        result = self.obj.get_wordposition_in_percent_in_sentences('ばかり', 0)
        self.assertEqual(len(self.obj.tokens_per_sentence), len(result))
        for i in result:
            self.assertEqual(1, len(i))
            self.assertGreater(100, max(i))
            self.assertGreater(min(i), 0)
        result = self.obj.get_wordposition_in_percent_in_sentences('EOS', 0)
        for i in result:
            self.assertEqual(i[0], 100.0)
        result = self.obj.get_wordposition_in_percent_in_sentences('で', 0)
        self.assertEqual(max([len(x) for x in result]), 3, "で kommt max. 3x in einem Satz vor")
        result = self.obj.get_wordposition_in_percent_in_sentences('、', 0)
        self.assertEqual(max([len(x) for x in result]), 4, "、 kommt max. 4x in einem Satz vor")

    def test_get_matrix_with_position_absolut_for_wordlist(self):
        self.obj.create_from_matrix(self.matrix)
        words = ['ばかり', 'なく', 'で', 'は', '、', '。']
        result2 = self.obj.get_matrix_with_position_absolut_for_wordlist(words, 0)
        expected2 = [[[4], [9], [9], [6], [12], [26], [9], [10], [10]],
                     [[6], [12], [11], [8], [14], [], [11], [12], [12]],
                     [[5], [10], [10], [7], [10, 13, 23], [34], [7, 10, 33], [11], [11, 15]],
                     [[2], [11], [1], [1], [3], [1], [1], [], [2, 7]],
                     [[7], [6, 13], [2, 12], [9, 16], [4, 15], [2, 17, 28, 35], [12, 28], [13], [8, 13]],
                     [[18], [18], [24], [22], [28], [45], [35], [30], [23]]]
        self.assertEqual(len(result2), len(words))
        self.assertEqual(result2, expected2)
        for i, tmp in enumerate(words):
            self.assertEqual(len(result2[i]), self.obj.max_y)

    def test_get_matrix_with_position_in_percent_for_wordlist(self):
        self.obj.create_from_matrix(self.matrix)
        words = ['ばかり', 'なく', 'で', 'は', '、', '。']
        result = self.obj.get_matrix_with_position_in_percent_for_wordlist(words, 0)
        expected_first_line = self.obj.get_wordposition_in_percent_in_sentences('ばかり', 0)
        self.assertEqual(expected_first_line, result[0])

    def test_map_every_element_recursive(self):
        self.obj.create_from_matrix(self.matrix)

        def foo(a):
            def fuu(b): return a * b
            return fuu

        liste = []
        result = self.obj.map_every_element_recursive(foo(2), liste)
        self.assertEqual(result, [])
        liste = [[[1], 10], [2], []]
        result = self.obj.map_every_element_recursive(foo(2), liste)
        self.assertEqual(result, [[[2], 20], [4], []])

    def test_verschiebe_dimension_der_matrix(self):
        import numpy as np
        """Urspruengliche Matrix = Matrix(9x47x30)
        Ergebnis Matrix = Matrix(47,30,9). Siehe auch Erklaerung in Funktionsdefinition"""
        self.obj.create_from_matrix(self.matrix)
        result = self.obj.verschiebe_dimension_der_matrix()
        self.assertEqual(len(result), self.obj.max_x)
        for y in result:
            self.assertEqual(len(y), self.obj.infos_per_token)
            for x in y:
                self.assertEqual(len(x), self.obj.max_y)
        # Die gleiche Funktionalitaet mit NumPy transpose()
        array = self.obj.as_numpy_array()
        new_matrix = np.transpose(array, (1, 2, 0))
        self.assertEqual(47, len(new_matrix))
        self.assertEqual(30, len(new_matrix[0]))
        self.assertEqual(9, len(new_matrix[0][0]))

    def test_center_matrix_at_positions(self):
        self.obj.create_from_matrix(self.matrix)
        result = self.obj.center_matrix_at_positions([[1]] * self.obj.number_of_sentences)
        self.assertEqual(len(result), self.obj.max_y)
        self.assertEqual(result[0][0][0], "に")
        self.assertEqual(result[1][0][0], 'はたん')
        self.assertEqual(result[-1][0][0], 'さん')
        result = self.obj.center_matrix_at_positions([[]] * self.obj.number_of_sentences)
        self.assertEqual(len(result), self.obj.max_y)
        self.assertFalse(result[0])
        self.assertFalse(result[1])
        self.assertFalse(result[-1])
        result = self.obj.center_matrix_at_positions([[1]] * self.obj.number_of_sentences, direction=0)
        self.assertEqual(len(result), self.obj.max_y)
        self.assertEqual(result[0][1][0], "に")
        self.assertEqual(result[1][1][0], 'はたん')
        self.assertEqual(result[-1][1][0], 'さん')
        [self.assertEqual(len(sentence), 2) for sentence in result]
        result = self.obj.center_matrix_at_positions([[1]] * self.obj.number_of_sentences, direction=-1)
        self.assertEqual(len(result), self.obj.max_y)
        self.assertEqual(result[0][0][0], "に")
        self.assertEqual(result[1][0][0], 'はたん')
        self.assertEqual(result[-1][0][0], 'さん')
        [self.assertEqual(len(sentence), 2) for sentence in result]

    def test_center_matrix_at_word(self):
        # TODO Siehe self.fail
        # self.fail(
        #     "Bisher nur fuer die erste Ebene Programmiert. Siehe Zuweisung in Funktion" +
        #     " <positions = self.getPositionsFor(word,0)>")
        self.obj.create_from_matrix(self.matrix)
        result = self.obj.center_matrix_at_word("ばかり")
        self.assertEqual(len(result), self.obj.max_y)
        [self.assertEqual(sentence[0][0], "ばかり") for sentence in result]
        [self.assertEqual(sentence[0][1], '助詞') for sentence in result]
        [self.assertIn(sentence[1][0], ['で', 'か']) for sentence in result]
        self.assertGreater(len(set([sentence[4][0] for sentence in result])), 3)
        result = self.obj.center_matrix_at_word("なく")

    def test_sort_list_of_keywords_by_field(self):
        self.obj.create_from_matrix(self.matrix)
        self.obj.execute_count_vectorizer_on_info_pos(0)
        dic = self.obj.create_dict_of_all_word_with_binary_count_quote()
        result = self.obj.sort_list_of_keywords_by_field(dic, 'binary')
        self.assertEqual(dic[result[0]]['binary'], 9)
        self.assertEqual(dic[result[-1]]['binary'], 1)
        result = self.obj.sort_list_of_keywords_by_field(dic, 'binary', reverse=False)
        self.assertEqual(dic[result[0]]['binary'], 1)
        self.assertEqual(dic[result[-1]]['binary'], 9)

    # TODO Bis hierhin gekommen

    def test_count_elements(self):
        assert True

    def test_rotate_sentence_numpy(self):
        assert True

    def test_rotate_matrix(self):
        assert True

    def test_join_tokens_at_same_column_in_sentence(self):
        assert True

    def test_create_frequenze_dict_for_every_row_in_matrix(self):
        assert True

    def test_calculate_position_index(self):
        assert True

    def test_calculate_position_index_in_percent(self):
        assert True

    def test_create_distance_matrix(self):
        assert True

    def test_get_sub_sentence(self):
        assert True

    def test_find_main_element(self):
        assert True
