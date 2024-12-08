from __future__ import annotations
import numpy
import sklearn.feature_extraction.text as skl
from typing import Callable, TypeVar, TYPE_CHECKING

from src.parser.NLPMecabParser import MecabSatzmatrix
import src.NLPMatrixFunctions as dmf

if TYPE_CHECKING:
    from src.parser.NLPMecabParser import MecabParser


T = TypeVar('T')


MecabTextmatrix = list[MecabSatzmatrix]
""" Also eine 3d-Matrix mit matrix_des_textes[satz_des_textes][word_im_satz][pos_des_wortes]
Die Wortart des dritten Wortes des 4. Satzes =  matrix[3][2][MecabPOS.WORTART.value]
"""


class NLPMecab:
    def __init__(self):
        self.matrix = []
        self.max_y = 0
        self.max_x = 0
        self.number_of_sentences = 0
        self.tokens_per_sentence = []
        self.infos_per_token = 30
        self.count_vectorizer = False
        self.count_vectorizer_result = False

    @staticmethod
    def build_matrix_from_list_of_sentences(tokenizer: Callable[[str], MecabParser],
                                            list_of_sentences: list[str]) -> MecabTextmatrix:
        """Jeder Satz wird in eine MecabSatzmatrix umgewandelt.
            Ergebnis ist text_matrix[satz_des_txtes][woerter/token_im_satz][pos_des_worts/tokens]
            tokenizer muss die Funktion as_matrix() -> MecabSatzmatrix implementieren"""
        return [tokenizer(satz).as_matrix() for satz in list_of_sentences]

    def create_from_matrix(self, matrix) -> NLPMecab:
        self.matrix = matrix
        self.max_y = len(matrix)
        self.max_x = self.get_max_x()
        self.tokens_per_sentence = self.number_of_tokens_per_sentence()
        self.number_of_sentences = len(self.tokens_per_sentence)
        return self

    def number_of_tokens_per_sentence(self) -> list[int]:
        """Ermittelt die Anzahl der Woerte/Tokens in pro Satz und gibt das Ergebnis als Liste zurueck."""
        return [len(row) for row in self.matrix] if self.matrix else [0]

    def get_max_x(self) -> int:
        """Die maximale Anzahl an Woertern/Token in einem Satz des Textes"""
        return max(self.number_of_tokens_per_sentence())

    def max_number_of_tokens_per_sentence(self) -> int:
        """Die Anzahl der Woert/Tokens des Satzes mit den meisten Woertern/Token"""
        return self.get_max_x()

    def get_mecab_info_at_position_per_sentence(self, info_pos: int = 0) -> list[list[str]]:
        """Ersetze die MecabSatzmatrix der Saetz (text_matrix[satz]) durch den Wert des Tokens an Position info_pos.
        Zum Beispiel:
            f(MecabPOS.WORT.value) -> list[satz][WOERTER]
            f(MecabPOS.WORTART.value) -> list[satz][WORTARTEN] : Dann koennte man die Anzahl der Wortarten ermittteln"""
        return [[token[info_pos] for token in sentence] for sentence in self.matrix]

    def trim_sentence_to_length(self, sentence, new_length) -> MecabSatzmatrix:
        """Erweiter/Reduziere die Anzahl der Woerter/Tokens des Satzes matrix[tokens][tokeninfos] auf new_length,
        damit eine einheitliche Matrix, zum Beispiel:
            f(satz, 50) -> satz[50 * Tokens][30 * MecabDaten]
        entsteht, die in ein NumpyArray umgewandelt werden kann. Kuerzere Saetze werden einfach durch Wiederholung
        des letzten Tokens auf die entsprechende Laenge gebracht. """
        last_token = sentence[-1]
        return sentence[:new_length] if new_length < len(sentence) else sentence + ([last_token] *
                                                                                    (new_length - len(sentence)))

    def as_numpy_array(self) -> numpy.ndarray:
        """Wandel meine Matrix in NumPy-Array um. Alle Saetze werden auf die Laenge des laengsten Satzes, also max_x,
        verlaengert mit trim_sentence_to_length(), damit die Matrix auch von NumPy akzeptiert wird."""
        return numpy.array([self.trim_sentence_to_length(sentence, self.max_x) for sentence in self.matrix])

    def get_vocabulary(self) -> dict[str, int]:
        """Liefert ein Dictionary mit den Woertern als Schluessel und den entsprechenden Werten des CountVectorizer"""
        return self.count_vectorizer.vocabulary_ if self.count_vectorizer else {}

    def execute_count_vectorizer_on_info_pos(self, info_pos: int = 0) -> NLPMecab:
        """Der Vectorizer weist jedem Wort eine Zahl zu."""
        def dummy(token): return token
        sentences_with_words = self.get_mecab_info_at_position_per_sentence(info_pos)
        self.count_vectorizer = skl.CountVectorizer(analyzer="word",
                                                    tokenizer=dummy,
                                                    preprocessor=dummy,
                                                    token_pattern=None)
        self.count_vectorizer_result = self.count_vectorizer.fit_transform(sentences_with_words)
        return self

    def get_number_of_occurence_per_sentence_for_word(self, word: str) -> numpy.ndarray:
        """Liefert ein NumPy-Array mit der Haufigkeit von word in jedem der Saetze.
        Wenn word in keinem der Saetze vorkommt, dann  [0] * anzahl_der_saetze."""
        index = self.count_vectorizer.vocabulary_.get(word)
        return self.count_vectorizer_result.toarray()[:, index]

    def create_binary_count_quote(self, word: str) -> dict[str, int]:
        """Liefer ein dicitonary mit folgenden drei Eintraegen:
            'binary': Anzahl der Saetze, in den das word vorkommt.
            'count': Gesamtzahl der Vorkommen des Wortes im Text.
            'quote': Setzt die binary ins Verhaeltnis zu count.
                Ein hoher quoten-Wert zeigt an, dass das Wort wichtig ist. Zum Beipiel in Beispielsaetzen zu einer
                bestimmten Grammatik. Ein Wort, das in jedem von 9 Saetzen einmal vorkommt
                ist wertvoller als ein Wort das in allen 9 Saetzen zweimal vorkommt.
                Ausserdem sollte der binary-Wert so dicht wie moeglich an der Gesamtzahl der Saetze liegen."""
        values = list(self.get_number_of_occurence_per_sentence_for_word(word))
        return {'binary': len(values)-values.count(0),
                'count': sum(values),
                'quote': (len(values)-values.count(0))/sum(values)}

    def create_dict_of_all_word_with_binary_count_quote(self) -> dict[str, dict[str:int]]:
        """Erstelle ein dict, in dem jedem Wort des CountVectorizers seine BinaryCountQuote zugewiesen wird"""
        word_list = self.get_vocabulary()
        return {word: self.create_binary_count_quote(word) for word in word_list}

    def filter_by_binary_count_quote(self, decider: Callable[[int], bool], key: str):
        """Filter das dict mit allen Wortern und ihrer BinaryCountQuote nach einem der drei Werte
        Zum Beispiel:
            f(lambda value: value >= 6, 'binary') -> Alle Woerter deren binary-Wert >= 6 ist"""
        return {word: value for word, value
                in self.create_dict_of_all_word_with_binary_count_quote().items() if decider(value[key])}

    def get_wordposition_absolute_in_sentences(self, word, info_pos: int = 0) -> list[list[int]]:
        """Liefert die Positionen des Wortes als Liste innerhalb der Saetze. info_pos ist die Position innerhalb
        des MecabTokenliste. Also info_pos = 2 liefert die Postitionen der Wortarten innerhalb des Satze.
        Kein Vorkommen -> []
        Einmaliges vorkommen -> [int]
        zweimaliges Vorkommen -> [int, int]
        Zum Beipiel:
            f('EOS', 0) sollte fuer jeden Satz eine Liste mit einem Element(x = len(satz), also Letztes) liefern.
            f('。', 0) sollte in den meisten Faellen die vorletzte Position liefern.
            f('名詞', 1) liefert die Postionen innerhalb der Saetze an denen ein Substantiv steht"""
        def get_pos_in_this_sentence(satzmatrix: list[list[str]], string: str, pos: int):
            return [index for index, tmp in enumerate(satzmatrix) if satzmatrix[index][pos] == string]
        return [get_pos_in_this_sentence(mecab_satzmatrix, word, info_pos) for mecab_satzmatrix in self.matrix]

    def get_wordposition_in_percent_in_sentences(self, word, info_pos: int = 0) -> list[list[float | int]]:
        """Liefert die Position innerhalb der Saetze nicht absolut, wie die Funktion
        get_get_wordposition_absolut_in_sentences(), sondern in Prozent. So kann man besser erkennen, ob ein
        Wort eher am Anfang, in der Mitte oder am Ende vorkommt."""
        def get_pos_in_this_sentence(satzmatrix: list[list[str]], string: str, pos: int):
            return [index for index, tmp in enumerate(satzmatrix) if satzmatrix[index][pos] == string]

        def calculate_percentage(positions, list_with_maxvalues):
            return list(map(lambda a, b: [elem * 100 / (b-1) if b > 1 else 0 for elem in a],
                            positions, list_with_maxvalues))
        return calculate_percentage(
            [get_pos_in_this_sentence(sentence, word, info_pos) for sentence in self.matrix],
            self.tokens_per_sentence)

    def get_matrix_with_position_absolut_for_wordlist(self,
                                                      list_of_words: list[str],
                                                      info_pos: int = 0) -> list[list[list[int]]]:
        """Ersetzt jedes Wort der Wortliste durch das Ergebniss von get_wordposition_absolute_in_sentences()
        Ein Liste mit 5 Woertern und einem Text mit neun Saetzen ergibt das eine matrix[5][9][pos]"""
        return [self.get_wordposition_absolute_in_sentences(word, info_pos) for word in list_of_words]

    def get_matrix_with_position_in_percent_for_wordlist(self,
                                                         list_of_words: list[str],
                                                         info_pos: int = 0) -> list[list[list[int | float]]]:
        """Ersetzt jedes Wort der Wortliste durch das Ergebniss von get_wordposition_in_percent_in_sentences()
        Ein Liste mit 5 Woertern und einem Text mit neun Saetzen ergibt eine matrix[5][9][pos]"""
        return [self.get_wordposition_in_percent_in_sentences(word, info_pos) for word in list_of_words]

    def map_every_element_recursive(self, map_func: Callable[[T], T], my_list: list[T]):
        """Fuehre die myFunc auf jedes Element in der Liste/Matrix aus."""
        # TODO Noch ohne Verwendungszweck
        # TODO Die Funktion hat nichts mit der Klasse zu tun und koennte/sollte in ein util-Modul
        # TODO Der rekursive Aufruf koennte bei grossen Matrizen zu Ueberaulf fuehren
        return list(map(lambda elem:
                        map_func(elem) if type(elem) is not list else self.map_every_element_recursive(map_func, elem),
                        my_list)) if my_list else []

    def verschiebe_dimension_der_matrix(self):
        # TODO alter Name: concat_matrix_on_axis()
        """Dreht die Matrix von Matrix(9x47x30) auf Matrix(47x30x9).
        Bei 9 Saetzen und einer maximalen Satzlaenge von 47 Woerten ist die
            getrimmte Matrix = Matrix(9x47x30)matrix[y][x][z].
                bildlich gesprochen mit einem Buch:
                    von links nach rechts stehen 47 Woerter
                    von oben nach unten stehen 9 Saetze.
                    auf den folgenden 30 Seiten steht an der gleichen Position wie auf Seite 1 die mecab_info
        Das Ergbnis dieser Funktion ist Matrix(47x30x9)
                bildlich gesprochen mit einem Buch:
                    von links nach rechts stehen die mecab_infos
                    von oben nach unten stehen die 47 Woerter eines Satzes.
                    auf den folgenden 9 Seiten steht an der gleichen Position wie auf Seite 1
                        das naechste Wort des Satzes
                Um den ersten Satz zu lesen, muesste man also auf der ersten Seite die erste Spalte von
                    oben nach unten lesen. matrix[[0-47]][0][0]
            Jetzt liegen die Information fuer jeden Satz in der z-Dimension uebereinander.
            Um eine bestimmte Grammatik kann man die 1. Dimension auf das Wort zentrieren, dann befindet sich in
            der 3. Dimension das gleiche Wort.

        Das selbe Ergebnis laesst sich auch mit NumPy-Transpose erreichen
            new_matrix = np.transpose(array, (1, 2, 0)) -> Matrix(47x30x9)
            new_matrix = np.transpose(array, (2, 0, 1)) -> Matrix(30x9x47)
            new_matrix = np.transpose(array, (2, 1, 0)) -> Matrix(30x47x9)
        """
        matrix = self.as_numpy_array()
        return [[matrix[:, x][:, z] for z in range(self.infos_per_token)] for x in range(self.max_x)]

    # TODO Bis hierhin gekommen.

    def sortKeywortsByField(self, dic, feldName, reverse = True):
        # TODO Kein Test
        sortedList = sorted(dic.items(), key=lambda x: x[1][feldName], reverse=reverse)
        return [elem[0] for elem in sortedList]

    def countElements(self,liste):
        ### Noch ohne Verwendung
        return {element: list(liste).count(element) for element in set(liste)}

    def rotateSentenceNumpy(self, sentence):
        ##### Noch keine Verwendung gefunden
        return numpy.rot90(sentence, k=0, axes=(1, 0))

    def centerMatrixAtPositions(self, listOfPositions: list[list[int]], direction = 1) -> list[list[list[str]]]:
        # wenn direction > 0, dann von position bis Ende, wenn direction < 0 dann von Position zum Anfang
        # bei direction -1 werden die Saetze umgedreht, so dass der Anfang des Satze sentence[-1] ist
        if len(listOfPositions) != self.number_of_sentences: raise NameError(f"Number of elements in List doesn't match the number of sentences!")
        if direction > 0: return [sentence[listOfPositions[index][0]:] if listOfPositions[index] else [] for index, sentence in enumerate(self.matrix)]
        if direction == 0: return [sentence[:listOfPositions[index][0]+1] if listOfPositions[index] else [] for index, sentence in enumerate(self.matrix)]
        if direction < 0: return [list(reversed(sentence[:listOfPositions[index][0]+1])) if listOfPositions[index] else [] for index, sentence in enumerate(self.matrix)]

    def centerMatrixAtWord(self, word):
        # Schreibe noch eine Funktion, die nicht nach Wort, sondern einer Liste mit einer Positionsangabe fuer jeden Satz arbeitet
        positions = self.get_wordposition_absolute_in_sentences(word, 0)
        return [sentence[positions[index][0]:] if positions[index] else [] for index, sentence in enumerate(self.matrix)]

    def rotateMatrix(self, matrix: list[list[list[int]]], placeHolder=None):
        maximum = max([len(x) for x in matrix])
        return [[row[i] if i < len(row) else placeHolder for row in matrix] for i in range(maximum)]

    def joinTokensAtSameColumnInSentence(self, matrix: list[list[list[str]]]) -> list[list[str]]:
        def withoutFirstColumn(sentenceMatrix): return list(map(lambda a: a[1:], sentenceMatrix))
        def joinTokensAtFirstColumnInSentence(sentenceMatrix):
            return [[y[i] for y in [x[0] for x in sentenceMatrix]] for i, tmp in enumerate(sentenceMatrix[0][1])]
        ####### Diese 15 kann bestimmt geloescht werden?????
        if len(matrix[0]) > 15:
            return joinTokensAtFirstColumnInSentence(matrix) + \
                    self.joinTokensAtSameColumnInSentence(withoutFirstColumn(matrix))
        else: return []

    def createFrequenzeDictForEveryRowInMatrix(self, matrix) -> list[dict[str:int]]:
        return [dmf.listAsFrequenzyDict(row) for row in matrix]

    def calculatePositionIndex(self, listOfPosition):
        binary = sum(map(lambda count: 1 if count else 0, listOfPosition))
        minimalPosCount = sum(map(lambda pos: min(pos) if pos else 0,listOfPosition))
        return minimalPosCount/binary if binary != 0 else False

    def calculatePositionIndexInPercent(self, listOfPosition):
        binary = sum(map(lambda count: 1 if count else 0, listOfPosition))
        minimalPosCount = sum(map(lambda pos: min(pos) if pos else 0,listOfPosition))
        return minimalPosCount/binary if binary != 0 else False

    def createDistanceMatrix(self,words,infoPos):
        positions = [self.get_wordposition_absolute_in_sentences(word, 0) for word in words]
        def createFullMatrix(pos):
            combinations = numpy.multiply.reduce([len(sen) for sen in pos if sen])
            return [elem * (combinations//len(elem)) if elem else [1000] * combinations for elem in pos]
        def subtract(array):
            #numpy.subtract.outer(elem[0],elem)
            return [numpy.subtract.outer(elem,array)[0] for elem in array]
        return subtract(createFullMatrix(self.rotateMatrix(positions)[3]))

    def getSubSentence(self, matrix: list[list[list[int]]], positionSlice: slice) -> list[list[list[int]]]:
        return [sentence[positionSlice] for sentence in matrix]

    def findMainElement(self):
        """Fuer das Hauptelement gilt:
                binary !< number_of_sentences
                binary == number_of_sentences evt. groesser
                binary == count bzw.

        :return:
        """