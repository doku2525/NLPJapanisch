from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
import MeCab
from typing import cast


class MecabPOS(Enum):
    WORT = 0
    WORTART = 1


MecabSatzmatrix = list[list[str]]


@dataclass(frozen=True)
class MecabParser:
    source: str = field(default_factory=str)
    mecabstring: str = None
    normale_token_laenge: int = 30
    _cached_matrix: list[MecabMatrix] = field(default_factory=list)

    @classmethod
    def parseText(cls, source) -> cls:
        """ Wandle mit Hilfe des Mecab.Tagger() den in source gespeicherten Text
            in einen mecabstring um. Mecabstring == pro Zeile ein Wort mit den Mecabdaten"""
        tagger = MeCab.Tagger()
        return cls(source=source,
                   mecabstring=tagger.parse(source))

    def as_matrix(self, benutze_cache: bool = True) -> MecabSatzmatrix:
        """Trenne zuerst das Wort von den Daten, die mit Tabulator getrennt sind.
            Dann Trenne die Daten an den Kommas mit Hilfe der Funktion splitte_an_kommas().
            Bei Eintraegen mit Anfuehrungszeichen kommt es zu Fehlern und es wird auch an den Kommas innerhalb
            der Anfuehrungszeichen getrennt. Dies wird nach dem Trennen durch die Funktion
            korrigiere_komma() korrigiert.
            Erzeugt die 2. Dimension der Matrix.
            """

        def korrigiere_komma(sentence: list[str]) -> list[str]:
            def repairFun(liste: list[str]) -> list[str]:
                return [liste[0] + ',' + liste[1]] + liste[2:]
            if not sentence:        # Liste durchgearbeitet, beende Rekursion.
                return []
            # Gerade Zahl an Anfuehrungszeichen, also muss nichts repariert werden
            if sentence[0].count('"') % 2 == 0:
                return [sentence[0]] + korrigiere_komma(sentence[1:])
            # Ungerade Anzahl an Anfuehrungszeichen, also rufe repairFun solange auf bis die Zahl wieder gerade ist.
            return korrigiere_komma(repairFun(sentence))

        def splitte_an_kommas(zeile: list[str, str]) -> list[str]:
            return [zeile[0]] + zeile[1].split(',') if len(zeile) == 2 else zeile

        def normalisiere_auf_dreizig_elemente(token: list[str]) -> list[str]:
            if token[0] == 'EOS':
                return ['EOS'] * 30
            if len(token) == 7 and token[2] == '数詞':
                return token + ['']*(30-len(token))
            if len(token) == 7 and token[2] == '普通名詞':
                return token + ['']*(30-len(token))
            if len(token) == 7 and token[1] == '感動詞':
                return token + ['']*(30-len(token))
            raise NameError(f'Fehler beim Erstellen der Matrix! Unbekannter Typ {token}')

        if self._cached_matrix and benutze_cache:
            return self._cached_matrix[0]
        # Splitte die Strings in Zeilen (\n) am Tabulator und dann an den Kommas mit Funktion konverterA
        result_list = map(splitte_an_kommas, [row.split('\t') for row in self.mecabstring.strip().split('\n')])
        # Repariere das Problem mit Kommas innerhalb von Anfuehrungszeichen
        repaired = map(lambda a: a if len(a) == 30 else korrigiere_komma(a), result_list)
        # Teste auf Elemente, die weniger als 30 Elemente haben und erweiter sie auf 30 Elemente
        self._cached_matrix.append(
            list(map(lambda a: normalisiere_auf_dreizig_elemente(a) if len(a) != 30 else a, repaired)))
        return self._cached_matrix[0]

    def as_wordlist_for_count_vectorizer(self) -> list[str]:
        """Liefert Liste mit den Woertern, also position 0, des Objekts.
        Liste kann dann in CountVectorizer benutzt werden"""
        return self.get_position_from_matrix(MecabPOS.WORT.value)

    @classmethod
    def tokenizer_for_count_vectorizer(cls, string) -> list[str]:
        """Liefert Liste mit den Woertern, also position 0, des Objekts.
        Liste kann dann in CountVectorizer benutzt werden.
        Die Funktion as_wordlist_for_count_vectorizer() als Klassenmethode implementiert."""
        return cls.parseText(string).get_position_from_matrix(MecabPOS.WORT.value)

    def get_position_from_matrix(self, position: int) -> list[str]:
        """Position gleich die Spalte der Matrix."""
        return [row[position] for row in self.as_matrix() if row[0] != 'EOS']

    def get_positionwindow_from_matrix(self, window: slice) -> list[list[str]]:
        """Positionsfenster liefert eine Matrix mit mehreren Spalten der Matrix."""
        return [row[window] for row in self.as_matrix() if row[0] != 'EOS']
