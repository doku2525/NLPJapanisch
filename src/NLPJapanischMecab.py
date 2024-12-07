import numpy
import sklearn.feature_extraction.text as skl
import NLPMatrixFunctions as dmf


class NLPMecab:
    def __init__(self):
        self.matrix = []
        self.maxY = 0
        self.maxX = 0
        self.number_of_sentences = 0
        self.tokens_per_sentence = []
        self.infos_per_token = 30
        self.count_vectorizer = False
        self.count_vectorizer_result = False

    def loadMatrix(self, matrix):
        self.matrix = matrix
        self.maxY = len(matrix)
        self.maxX = self.getMaxX()
        self.tokens_per_sentence = self.tokensPerSentence()
        self.number_of_sentences = len(self.tokens_per_sentence)
    def getMaxX(self):
        return max(self.tokensPerSentence())
    def buildMatrixFromList(tokenizer, listOfSentences):
        return [tokenizer(x) for x in listOfSentences]
    def tokensPerSentence(self):
        return [len(row) for row in self.matrix] if self.matrix else [0]
    def getMecabInfoAtPositionPerSentence(self,position):
        return [[token[position] for token in sentence] for sentence in self.matrix]
    def trimSentencesToLength(self,sentence,newLength):
        lastToken = sentence[-1]
        if newLength < len(sentence): return sentence[:newLength]
        else: return sentence + ([lastToken] * (newLength - len(sentence)))
    def asNumpyArray(self):
        def recFunction(sentence):
            if not sentence: return []
            else: return [self.trimSentencesToLength(sentence[0], self.maxX)] + recFunction(sentence[1:])
        return numpy.array(recFunction(self.matrix))

    def getVocabulary(self):
        return self.count_vectorizer.vocabulary_ if self.count_vectorizer else {}

    def getCountArrayFor(self, word):
        index = self.count_vectorizer.vocabulary_.get(word)
        return self.count_vectorizer_result.toarray()[:, index]

    def getCountValuesFor(self, word):
        values = list(self.getCountArrayFor(word))
        return {'binary': len(values)-values.count(0),
                'count': sum(values),
                'quote': (len(values)-values.count(0))/sum(values)}

    def countWords(self):
        wordList = self.getVocabulary()
        return {word:self.getCountValuesFor(word) for word in wordList}

    def selectAllWithCount(self, option, grenze, strikt):
        wordList = self.countWords()
        decider = lambda a: a >= grenze
        return {word:value for word, value in wordList.items() if decider(value[option])}

    def getPositionsFor(self, word, infoPos):
        def getPosInThisSentence(matrix, string, pos): return [index for index, tmp in enumerate(matrix) if matrix[index][pos] == string]
        return [getPosInThisSentence(sentence, word, infoPos) for sentence in self.matrix]

    def getPositionsForInPercent(self, word, infoPos):
        def getPosInThisSentence(matrix, string, pos): return [index for index, tmp in enumerate(matrix) if matrix[index][pos] == string]
        def calculatePercentage(positions,listWithMaxvalues): return list(map(lambda a, b: [elem * 100 / (b-1) if b>1 else 0 for elem in a], positions, listWithMaxvalues))
        return calculatePercentage(
            [getPosInThisSentence(sentence, word, infoPos) for sentence in self.matrix],
            self.tokens_per_sentence)

    def getMatrixWithPositionForWordlist(self, listOfWords, infoPos):
        return [self.getPositionsFor(word, 0) for word in listOfWords]

    def getMatrixWithPositionForWordlistInPercent(self, listOfWords, infoPos):
        return [self.getPositionsForInPercent(word, 0) for word in listOfWords]

    def mapEveryElementRecursive(self, myFunction, myList):
        #### Noch ohne Verwendungszweck
        if not myList: return []
        return list(map(lambda elem:
                        myFunction(elem) if type(elem) is not list else self.mapEveryElementRecursive(myFunction, elem),
            myList))

    def executeCountVectorizerOnInfoPos(self, position):
        def dummy(token): return token
        sentencesWithWords = self.getMecabInfoAtPositionPerSentence(position)
        self.count_vectorizer = skl.CountVectorizer(analyzer="word",
                                        tokenizer=dummy,
                                        preprocessor=dummy,
                                        token_pattern=None)
        self.count_vectorizer_result = self.count_vectorizer.fit_transform(sentencesWithWords)

    def sortKeywortsByField(self, dic, feldName, reverse = True):
        sortedList = sorted(dic.items(), key=lambda x: x[1][feldName], reverse=reverse)
        return [elem[0] for elem in sortedList]
    def concatMatrixOnAxis(self):
        ### Status = Test
        matrix = self.asNumpyArray()
        return [[matrix[:,x][:,z] for z in range(self.infos_per_token)] for x in range(self.maxX)]

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
        positions = self.getPositionsFor(word,0)
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
        positions = [self.getPositionsFor(word, 0) for word in words]
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