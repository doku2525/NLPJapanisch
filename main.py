import src.NLPMecab as nlp
from src.parser.NLPMecabParser import MecabParser


def ladeDaten(filename):
    with open(filename, "r") as f:
        result = f.read()
    return result.split()


def waehle_sprache(modulListe):
    print("Willkommen zum Doku-NLP-Projekt!")
    print("Waehle aus den verfuegbaren Sprachen:")
    for i, modul in enumerate(modulListe):
        print("\t ({}) {}\n".format(i, modul[1]))
    antwort = int(input("\nEingabe : "))
    return antwort if antwort in range(0, len(modulListe)+1) else quit


def print_auswahl(aktionen):
    print("Was soll ich tun?")
    for i, aktion in enumerate(aktionen):
        print("\t ({}) {}".format(i, aktion[0]))
    eingabe = int(input("Eingabe: "))
    if eingabe not in range(0,len(aktionen)):
        return falscheEingabe
    return aktionen[eingabe][1]


def pruefe_neues_objekt(obj):
    if not obj:
        return objekthistory_list
    if obj != objekthistory_list[-1]:
        print("Neues Objekt <<{}>>! Mit neuem Objekt fortfahren?".format(obj['name']))
        eingabe = input("j/n ").lower()
        if eingabe == 'j':
            return objekthistory_list + [obj]
        else:
            return objekthistory_list
    else:
        return objekthistory_list


def lade_japanisch(dateiname):
    print(dateiname)
    sentences = ladeDaten(dateiname)
    print("{} Saetze gefunden!".format(len(sentences)))
    print("Erstelle mein NLP Objekt.")
    obj = nlp.NLPMecab()
    matrix = nlp.NLPMecab.build_matrix_from_list_of_sentences(MecabParser.parseText, sentences)
    obj.create_from_matrix(matrix)
    return obj


def beende_programm():
    print("Beende Programm!")
    quit()


def zeigeWortzahl():
    objekt = objekthistory_list[-1]['objekt']
    objekt.execute_count_vectorizer_on_info_pos(aktuelle_ebene)
    result = objekt.create_dict_of_all_word_with_binary_count_quote()
    print(result)
    return False


def begrenze_auf_wortzahl():
    mein_objekt = objekthistory_list[-1]['objekt']
    prozent = int(input("Setze Grenze auf wieviel Prozent? (1-100) : "))
    mein_objekt.execute_count_vectorizer_on_info_pos(aktuelle_ebene)
#    result = objekt.selectAllWithCount('quote', objekt.number_of_sentences * prozent /100, False)
    result = mein_objekt.filter_by_binary_count_quote(lambda value: value >= prozent / 100, 'quote')
    print("{} Woerter gefunden.".format(len(result.keys())))
    maxWordLength = max([len(word) for word in list(result.keys())])
    sortedKeys = mein_objekt.sort_list_of_keywords_by_field(result,'binary')
    for index, i in enumerate( sortedKeys ):
        if mein_objekt.number_of_sentences * prozent /100 < result[i]['binary']:
            print("{} {}\tSaetze:{}  Gesamt:{}  jQuote:{}  PosAbsolut:{} Pos in %:{}".format(
                f"{index+1: >{3}}",
                f"{i: <{maxWordLength+2}}",
                f"{result[i]['binary']: >{2}}",
                f"{result[i]['count']: >{3}}",
                f"{result[i]['quote']*100:7.2f}",
                f"{mein_objekt.calculate_position_index(mein_objekt.get_wordposition_absolute_in_sentences(i,aktuelle_ebene)):6.2f}",
                f"{mein_objekt.calculate_position_index(mein_objekt.get_wordposition_in_percent_in_sentences(i, aktuelle_ebene)):6.2f}"))
    eingabe = input("Zeige PositionsMatrix fuer bestimmtes Wort oder Zurueck(0) ")
    if eingabe.isdigit():
        if int(eingabe) == 0: return False
        wordToSearch = sortedKeys[int(eingabe)-1]
    elif eingabe not in sortedKeys: begrenze_auf_wortzahl()
    else: wordToSearch = eingabe
    zeige_positions_matrix_fuer_wort(mein_objekt.get_wordposition_absolute_in_sentences(wordToSearch, aktuelle_ebene))
    begrenze_auf_wortzahl()
    return False


def zeige_wortzahl():
    mein_objekt = objekthistory_list[-1]['objekt']
    mein_objekt.execute_count_vectorizer_on_info_pos(aktuelle_ebene)
    result = mein_objekt.create_dict_of_all_word_with_binary_count_quote()
    print(result)
    return False


def print_mecab_ebene():
    obj = objekthistory_list[-1]['objekt']
    meine_eingabe = input("Welche Ebene soll ich zeigen? (0-29) : ")
#    obj.getMecabInfoAtPositionPerSentence(int(eingabe))
    for i in obj.get_mecab_info_at_position_per_sentence(int(meine_eingabe)):
        print(i)
    return False


def neue_matrix_fuer_wort() -> dict[str, nlp.NLPMecab]:
    wort = input("Welches Wort soll ich benutzen? ").strip()
    print("Erstelle Matrix fuer {}!".format(wort))
    obj = objekthistory_list[-1]['objekt']
    obj.execute_count_vectorizer_on_info_pos(aktuelle_ebene)
    neue_matrix = obj.center_matrix_at_word(wort)
    mein_objekt = nlp.NLPMecab()
    mein_objekt.create_from_matrix(neue_matrix)
    return {'name': wort, 'objekt': mein_objekt}


def neue_ebene() -> bool:
    global aktuelle_ebene
    nummer = input(f"Aendere Ebene von {aktuelle_ebene} auf: ")
    aktuelle_ebene = int(nummer)
    return False


def zeige_positions_matrix_fuer_wort(liste) -> None:
    for i in liste:
        print("  ", i)


""" 
*******************
Definiere die globalen Variablen
********************
"""
objekthistory_list = []
aktuelle_ebene = 0
meine_module = [
    ['jap', 'Japanisch mit Mecab', lade_japanisch]
]
meine_aktionen = [
    ['Beende Programm', beende_programm],
    ['Zeige Wortzahl.', zeige_wortzahl],
    ['Zeige haufigsten Woerter.', begrenze_auf_wortzahl],
    ['Zeige Text at Info Ebene', print_mecab_ebene],
    ['Neue Matrix an einem Wort zentriert', neue_matrix_fuer_wort],
    [f'Andere aktuelle Ebene (aktuell : {aktuelle_ebene})', neue_ebene]
]


if __name__ == '__main__':
    sprache = waehle_sprache(meine_module)
    print("Lade Testdaten!")
    objekt = meine_module[sprache][2]('data/testdata.{}'.format(meine_module[sprache][0]))
    objekthistory_list = objekthistory_list + [{'name': 'Gesamt', 'objekt': objekt}]
    print("MaxX = ", objekt.max_x, "  MaxY = ", objekt.max_y)
    print("\n")

    while True:
        eingabe = print_auswahl(meine_aktionen)
        objekthistory_list = pruefe_neues_objekt(eingabe())
