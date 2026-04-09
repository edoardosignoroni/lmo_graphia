"""
LOMBARD ORTHOGARPHY CONVERTER
version 0.1.0 alpha

ONLY BERGDUC TO IPA RULE-BASED CONVERSION
"""
import argparse

def get_next_char(word: str, index: int) -> str | None:
    if index + 1 < len(word):
        return word[index + 1]
    return None

def get_previous_char(word: str, index: int) -> str | None:
    if index - 1 >= 0:
        return word[index - 1]
    return None

def is_last(word: str, index: int) -> bool:
    return index == len(word) - 1

def is_first(word: str, index: int) -> bool:
    return index == 0

def is_vowel(char: str) -> bool:
    return char in "aeiouáàéèíìóòúù" if char else False

def is_consonant(char: str) -> bool:
    return char in "bcdfghjklmnpqrstvwxyz" if char else False   

def convert_text(text: str, source: str, target: str) -> str:
    """
    Convert the given text from BERGDUC orthography to NOL orthography.

    :param text: The input text in BERGDUC orthography.
    :param source: The source orthography.
    :param target: The target orthography.  
    :return: The converted text in NOL orthography.
    """
    if source == "BERGDUC" and target == "NOL":
        return bergduc_to_nol(text)
    elif source == "BERGDUC" and target == "IPA":
        return bergduc_to_ipa(text)
    elif source == "IPA" and target == "BERGDUC":
        return ipa_to_bergduc(text)
    else:
        raise ValueError(f"Conversion from {source} to {target} is not supported.")

def bergduc_to_ipa(text: str) -> str: # SOME RULES ARE REDUNDANT BECAUSE THEN THE GRAPHEMES WILL BE DIFFERENT
    """
    Convert the given text from BERGDUC orthography to IPA transcription.

    :param text: The input text in BERGDUC orthography.
    :return: The converted text in IPA transcription.
    """
    converted_text = []
    for word in text.split():
        converted_word = []
        print(f"Converting word: {word}")
        converted_char = ""
        skip_until = -1
        for i in range(len(word)):
            if i < skip_until:
                continue
            print(f"Processing character: {word[i]} at index {i}")
            char_idx = i
            char = word[i].lower()
            if char == "a":
                if get_next_char(word, char_idx) == "a":
                    converted_char = "a:"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    converted_char = "a"
            elif char == "à":
                if get_next_char(word, char_idx) == "a":
                    converted_char = "a:"
                    skip_until = i + 2 # skips the next char since it is already converted
                if is_last(word, char_idx):
                    converted_char = "a"
                else:
                    converted_char = "a"
            elif char == "b":
                if is_last(word, char_idx):
                    converted_char = "p"
                else:
                    converted_char = "b"
            elif char == "c":
                if get_next_char(word, char_idx) == "c":
                    converted_char = "tʃ"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    if get_next_char(word, char_idx) in ["e", "i", "é", "è"]:
                        if get_previous_char(word, char_idx) == "s":
                            converted_char = "" # handles sc
                        elif get_previous_char(word, char_idx) == "c" and is_last(word, char_idx - 1):
                            converted_char = "" # handles cc in the end of a word, which is pronounced as "k" in milanese but as "tʃ" in bergduc, thus we convert it to "" to avoid confusion and to signal that it is there
                        else:
                            converted_char = "tʃ"
                            # In sequences like "ci" before a vowel, i marks palatalization and should not become a separate j.
                            if get_next_char(word, char_idx) == "i" and is_vowel(get_next_char(word, char_idx + 1)):
                                skip_until = i + 2
                    else:
                        converted_char = "k"
            elif char == "d":
                if is_last(word, char_idx):
                    converted_char = "t"
                else:
                    converted_char = "d"
            elif char == "e":
                if get_next_char(word, char_idx) == "e":
                    converted_char = "e:"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    converted_char = "e"
            elif char == "è":
                if get_next_char(word, char_idx) == "e":
                    converted_char = "e:"
                    skip_until = i + 2 # skips the next char since it is already converted
                elif is_last(word, char_idx):
                    converted_char = "ɛ"
                else:
                    converted_char = "ɛ"
            elif char == "é":
                if get_next_char(word, char_idx) == "e":
                    converted_char = "e:"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    converted_char = "e"
            elif char == "f":
                converted_char = "f"
            elif char == "g":
                if get_next_char(word, char_idx) == "n":
                    converted_char = "ɲ"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    if get_next_char(word, char_idx) in ["e", "i", "é", "è", "í", "ì"]:
                        converted_char = "dʒ" # giornàda
                        skip_until = i + 2 # skips the next char since it is already converted
                    else:
                        converted_char = "g"
            elif char == "h":
                converted_char = "" # for testing and to know that it is there
            elif char == "i":
                if is_vowel(get_previous_char(word, char_idx)) and is_vowel(get_next_char(word, char_idx)):
                    converted_char = "j"
                elif is_consonant(get_previous_char(word, char_idx)) and is_vowel(get_next_char(word, char_idx)):
                    if get_previous_char(word, char_idx) == "g":
                        converted_char = "" # handles gi
                    else:
                        converted_char = "j"
                else:
                    converted_char = "i"
            elif char == "ì":
                if is_last(word, char_idx):
                    converted_char = "i"
                else:
                    converted_char = "i"
            elif char == "ï":
                converted_char = "i"
            elif char == "j":
                converted_char = "j"
            elif char == "k":
                converted_char = "k"
            elif char == "l":
                if is_last(word, char_idx) and get_previous_char(word, char_idx) in ["á", "à", "è", "é", "ì", "í", "ó", "ò", "ú", "ù"]:
                    converted_char = "l"
                else:
                    converted_char = "l" 
            elif char == "m":
                converted_char = "m"
            elif char == "n":
                if is_last(word, char_idx) and get_previous_char(word, char_idx) in ["á", "à", "è", "é", "ì", "í", "ó", "ò", "ú", "ù"]:
                    converted_char = "n"
                else:
                    converted_char = "n"
            elif char == "o":
                if is_last(word, char_idx):
                    converted_char = "o:" # sovraestesa DA CONTROLLARE
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    converted_char = "o"
            elif char == "ò":
                converted_char = "ɔ"
            elif char == "ó":
                converted_char = "o"
            elif char == "ö":
                converted_char = "ø" # to check the long version
            elif char == "p":
                converted_char = "p"
            elif char == "q":
                if get_next_char(word, char_idx) == "u":
                    converted_char = "kw"
                    skip_until = i + 2 # skips the next char since it is already converted
                else:
                    converted_char = "" # for testing and to know that it is there
            elif char == "r":
                converted_char = "r" 
            elif char == "s": # other rules for s are ambigous and dependent on pronunciation of milanese (thus signalling that NOL is milanese-based and biased)
                if is_last(word, char_idx) and get_next_char(word, char_idx) not in ["c", "g"]:
                    converted_char = "s"
                elif is_vowel(get_previous_char(word, char_idx)) and is_vowel(get_next_char(word, char_idx)) and get_next_char(word, char_idx) != "s":        
                    converted_char = "z"
                elif is_consonant(get_previous_char(word, char_idx)):
                    converted_char = "s"
                elif is_consonant(get_next_char(word, char_idx)):
                    if get_next_char(word, char_idx) == "c":
                        if get_next_char(word, char_idx + 1) in ["i", "e", "é", "è", "í", "ì"]: 
                            converted_char = "ʃ"
                            # In sequences like "sci" before a vowel, the "i" is part of the cluster and should be skipped.
                            if get_next_char(word, char_idx + 1) == "i" and is_vowel(get_next_char(word, char_idx + 2)):
                                skip_until = i + 3
                            else:
                                skip_until = i + 2
                        else:
                            converted_char = "sk"
                            skip_until = i + 2
                    else:
                        converted_char = "s"
                elif get_next_char(word, char_idx) == "-" and get_next_char(word, char_idx + 1) == "c":
                    converted_char = "stʃ"
                    skip_until = i + 3 # skips next two chars since they are already converted
                elif get_next_char(word, char_idx) == "g":
                    converted_char = "ʒ"
                    skip_until = i + 2
                else:
                    converted_char = "s"
            elif char == "t":
                if is_last(word, char_idx):
                    converted_char = "t"
                else:
                    converted_char = "t"
            elif char == "u":
                if is_vowel(get_previous_char(word, char_idx)) or is_vowel(get_next_char(word, char_idx)):
                    converted_char = "w"
                elif get_previous_char(word, char_idx) == "q":
                    converted_char = "w"
                else:
                    converted_char = "u"
            elif char == "ù":
                if is_last(word, char_idx):
                    converted_char = "u"
                else:
                    converted_char = "u"
            elif char == "ü":
                converted_char = "y"
            elif char == "v":
                if is_last(word, char_idx):
                    converted_char = "f"
                else:
                    converted_char = "v"
            elif char == "w":
                converted_char = "w"
            elif char == "x":
                converted_char = "ks"
            elif char == "y":
                converted_char = "i"
            elif char == "z":
                converted_char = "z"
            else:
                converted_char = char # for characters that are not in the BERGDUC orthography, we keep them as they are
            
            converted_word.append(converted_char)
        
        print(f"Converted word: {''.join(converted_word)}")
        
        converted_text.append("".join(converted_word))
    return " ".join(converted_text)

def ipa_to_bergduc(text: str) -> str:
    """
    Convert the given text from IPA transcription to BERGDUC orthography.

    Note: this is a best-effort inverse mapping; BERGDUC->IPA is lossy in
    several places, so some graphemic ambiguities cannot be perfectly restored.

    :param text: The input text in IPA transcription.
    :return: The converted text in BERGDUC orthography.
    """
    # Implementation of the conversion logic goes here
    pass

def nol_to_ipa(text: str) -> str:
    """
    Convert the given text from NOL orthography to IPA transcription.

    :param text: The input text in NOL orthography.
    :return: The converted text in IPA transcription.
    """
    # Implementation of the conversion logic goes here
    pass

def ipa_to_nol(text: str) -> str:
    """
    Convert the given text from IPA transcription to NOL orthography.

    Note: this is a best-effort inverse mapping; NOL->IPA is lossy in
    several places, so some graphemic ambiguities cannot be perfectly restored.

    :param text: The input text in IPA transcription.
    :return: The converted text in NOL orthography.
    """
    # Implementation of the conversion logic goes here
    pass

def bergduc_to_nol(text: str) -> str:
    """
    Convert the given text from BERGDUC orthography to NOL orthography.

    :param text: The input text in BERGDUC orthography.
    :return: The converted text in NOL orthography.
    """
    # Implementation of the conversion logic goes here
    pass

def nol_to_bergduc(text: str) -> str:
    """
    Convert the given text from NOL orthography to BERGDUC orthography.

    Note: this is a best-effort inverse mapping; BERGDUC->NOL is lossy in
    several places, so some graphemic ambiguities cannot be perfectly restored.

    :param text: The input text in NOL orthography.
    :return: The converted text in BERGDUC orthography.
    """
    # Implementation of the conversion logic goes here
    pass

def dictionary_lookup(word: str, source: str, target: str) -> str | None:
    """
    Look up the given word in a dictionary for the specified source and target orthographies.

    :param word: The word to look up.
    :param source: The source orthography.
    :param target: The target orthography.
    :return: The converted word if found in the dictionary, otherwise None.
    """
    # Implementation of the dictionary lookup logic goes here
    pass

def freq_lookup(word: str, source: str, target: str) -> str | None:
    """
    Look up the frequency of the given word in a frequency list for the specified source and target orthographies.

    :param word: The word to look up.
    :param source: The source orthography.
    :param target: The target orthography.
    :return: The frequency of the word if found in the frequency list, otherwise None.
    """
    # Implementation of the frequency lookup logic goes here
    pass

def char_probability_backoff(char: str, source: str, target: str) -> float | None:
    """
    Look up the probability of the given character in a character probability list for the specified source and target orthographies.

    :param char: The character to look up.
    :param source: The source orthography.
    :param target: The target orthography.
    :return: The probability of the character if found in the list, otherwise None.
    """
    # Implementation of the character probability lookup logic goes here
    pass

def parse_input_file(input_file: str) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        return f.readlines()

def main():
    parser = argparse.ArgumentParser(description="Convert text from BERGDUC orthography to NOL orthography.")
    parser.add_argument("--text", type=str, default="", help="The input text in BERGDUC orthography.")
    parser.add_argument("--input", type=str, help="The input file containing text in BERGDUC orthography.")
    parser.add_argument("--output", type=str, help="The output file to save the converted text in NOL orthography.")
    parser.add_argument("--source", type=str, default="BERGDUC", help="The source orthography (default: BERGDUC).")
    parser.add_argument("--target", type=str, default="NOL", help="The target orthography (default: NOL).")
    
    args = parser.parse_args()
    
    if args.input:
        text = parse_input_file(args.input)
        converted_text = "\n".join([convert_text(line, args.source, args.target) for line in text])
    else:        
        text = args.text
        converted_text = convert_text(text, args.source, args.target)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(converted_text)
    else:
        print(converted_text)

if __name__ == "__main__":
    main()