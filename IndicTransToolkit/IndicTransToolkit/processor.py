"""
Python version of the IndicProcessor class converted from Cython.
Methods preprocess_batch and postprocess_batch are exposed for external use.
All other methods are internal and use Python conventions.
"""

import regex as re
from tqdm import tqdm
from queue import Queue
from typing import List

# Importing Python objects since these libraries don't offer C-extensions
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


class IndicProcessor:
    def __init__(self, inference=True):
        """
        Constructor for IndicProcessor. Initializes all necessary components.
        """
        self.inference = inference

        ##############################
        # FLORES -> ISO CODES
        ##############################
        self._flores_codes = {
            "asm_Beng": "as",
            "awa_Deva": "hi",
            "ben_Beng": "bn",
            "bho_Deva": "hi",
            "brx_Deva": "hi",
            "doi_Deva": "hi",
            "eng_Latn": "en",
            "gom_Deva": "kK",
            "gon_Deva": "hi",
            "guj_Gujr": "gu",
            "hin_Deva": "hi",
            "hne_Deva": "hi",
            "kan_Knda": "kn",
            "kas_Arab": "ur",
            "kas_Deva": "hi",
            "kha_Latn": "en",
            "lus_Latn": "en",
            "mag_Deva": "hi",
            "mai_Deva": "hi",
            "mal_Mlym": "ml",
            "mar_Deva": "mr",
            "mni_Beng": "bn",
            "mni_Mtei": "hi",
            "npi_Deva": "ne",
            "ory_Orya": "or",
            "pan_Guru": "pa",
            "san_Deva": "hi",
            "sat_Olck": "or",
            "snd_Arab": "ur",
            "snd_Deva": "hi",
            "tam_Taml": "ta",
            "tel_Telu": "te",
            "urd_Arab": "ur",
            "unr_Deva": "hi",
        }

        ##############################
        # INDIC DIGIT TRANSLATION (str.translate)
        ##############################
        digits_dict = {
            "\u09e6": "0", "\u0ae6": "0", "\u0ce6": "0", "\u0966": "0",
            "\u0660": "0", "\uabf0": "0", "\u0b66": "0", "\u0a66": "0",
            "\u1c50": "0", "\u06f0": "0",

            "\u09e7": "1", "\u0ae7": "1", "\u0967": "1", "\u0ce7": "1",
            "\u06f1": "1", "\uabf1": "1", "\u0b67": "1", "\u0a67": "1",
            "\u1c51": "1", "\u0c67": "1",

            "\u09e8": "2", "\u0ae8": "2", "\u0968": "2", "\u0ce8": "2",
            "\u06f2": "2", "\uabf2": "2", "\u0b68": "2", "\u0a68": "2",
            "\u1c52": "2", "\u0c68": "2",

            "\u09e9": "3", "\u0ae9": "3", "\u0969": "3", "\u0ce9": "3",
            "\u06f3": "3", "\uabf3": "3", "\u0b69": "3", "\u0a69": "3",
            "\u1c53": "3", "\u0c69": "3",

            "\u09ea": "4", "\u0aea": "4", "\u096a": "4", "\u0cea": "4",
            "\u06f4": "4", "\uabf4": "4", "\u0b6a": "4", "\u0a6a": "4",
            "\u1c54": "4", "\u0c6a": "4",

            "\u09eb": "5", "\u0aeb": "5", "\u096b": "5", "\u0ceb": "5",
            "\u06f5": "5", "\uabf5": "5", "\u0b6b": "5", "\u0a6b": "5",
            "\u1c55": "5", "\u0c6b": "5",

            "\u09ec": "6", "\u0aec": "6", "\u096c": "6", "\u0cec": "6",
            "\u06f6": "6", "\uabf6": "6", "\u0b6c": "6", "\u0a6c": "6",
            "\u1c56": "6", "\u0c6c": "6",

            "\u09ed": "7", "\u0aed": "7", "\u096d": "7", "\u0ced": "7",
            "\u06f7": "7", "\uabf7": "7", "\u0b6d": "7", "\u0a6d": "7",
            "\u1c57": "7", "\u0c6d": "7",

            "\u09ee": "8", "\u0aee": "8", "\u096e": "8", "\u0cee": "8",
            "\u06f8": "8", "\uabf8": "8", "\u0b6e": "8", "\u0a6e": "8",
            "\u1c58": "8", "\u0c6e": "8",

            "\u09ef": "9", "\u0aef": "9", "\u096f": "9", "\u0cef": "9",
            "\u06f9": "9", "\uabf9": "9", "\u0b6f": "9", "\u0a6f": "9",
            "\u1c59": "9", "\u0c6f": "9",
        }
        self._digits_translation_table = {ord(k): v for k, v in digits_dict.items()}

        # Also map ASCII '0'-'9'
        for c in range(ord('0'), ord('9') + 1):
            self._digits_translation_table[c] = chr(c)

        ##############################
        # PLACEHOLDER MAP QUEUE
        ##############################
        self._placeholder_entity_maps = Queue()

        ##############################
        # MOSES (as Python objects)
        ##############################
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

        ##############################
        # TRANSLITERATOR (Python object)
        ##############################
        self._xliterator = UnicodeIndicTransliterator()

        ##############################
        # Precompiled Patterns
        ##############################
        self._MULTISPACE_REGEX = re.compile(r"[ ]{2,}")
        self._DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
        self._DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
        self._DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
        self._END_BRACKET_SPACE_PUNC_REGEX = re.compile(r"\) ([\.!:?;,])")

        self._URL_PATTERN = re.compile(
            r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"
        )
        self._NUMERAL_PATTERN = re.compile(
            r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
        )
        self._EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
        self._OTHER_PATTERN = re.compile(r"[A-Za-z0-9]*[#|@]\w+")

        # Combined punctuation replacements
        self._PUNC_REPLACEMENTS = [
            (re.compile(r"\r"), ""),
            (re.compile(r"\(\s*"), "("),
            (re.compile(r"\s*\)"), ")"),
            (re.compile(r"\s:\s?"), ":"),
            (re.compile(r"\s;\s?"), ";"),
            (re.compile(r"[`´‘‚’]"), "'"),
            (re.compile(r"[„“”«»]"), '"'),
        ]

        ##############################
        # INDIC NORMALIZER
        ##############################
        normalizer_factory = IndicNormalizerFactory()
        self._indic_normalizer = normalizer_factory.get_normalizer("hi")

    def preprocess_batch(self, inputs: List[str], language: str):
        """
        Preprocess a batch of texts.
        """
        processed_batch = []
        for sentence in inputs:
            processed_batch.append(self._preprocess(sentence, language))
        return processed_batch

    def postprocess_batch(self, inputs: List[str], language: str):
        """
        Postprocess a batch of texts.
        """
        processed_batch = []
        for sentence in inputs:
            processed_batch.append(self._postprocess(sentence, language))
        return processed_batch

    def _preprocess(self, sentence: str, language: str) -> str:
        """
        Core preprocessing logic for a single sentence.
        """
        if not sentence or sentence.strip() == "":
            return ""

        # Normalize digits by translation
        sentence = sentence.translate(self._digits_translation_table)

        # Normalize unicode punctuation to ASCII equivalents
        for pattern, replacement in self._PUNC_REPLACEMENTS:
            sentence = pattern.sub(replacement, sentence)

        # Remove multiple spaces
        sentence = self._MULTISPACE_REGEX.sub(" ", sentence).strip()

        # Normalize percent digit spacing
        sentence = self._DIGIT_SPACE_PERCENT.sub(r"\1%", sentence)

        # Normalize double quotes punctuation spacing
        sentence = self._DOUBLE_QUOT_PUNC.sub(r'"\1', sentence)

        # Remove NBSP between digits
        sentence = self._DIGIT_NBSP_DIGIT.sub(r"\1\2", sentence)

        # Fix spacing after end bracket
        sentence = self._END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", sentence)

        # Normalize Indic text (for Hindi here, could parameterize)
        sentence = self._indic_normalizer.normalize(sentence)

        # Tokenize sentence
        if language == "en":
            # English tokenization using Moses tokenizer
            sentence = self._en_tok.tokenize(sentence, return_str=True)
        else:
            # Indic tokenization
            sentence = " ".join(indic_tokenize.trivial_tokenize(sentence, lang=language))

        # Further normalize
        sentence = self._MULTISPACE_REGEX.sub(" ", sentence).strip()

        return sentence

    def _postprocess(self, sentence: str, language: str) -> str:
        """
        Core postprocessing logic for a single sentence.
        """
        if not sentence or sentence.strip() == "":
            return ""

        # Detokenize the sentence based on language
        if language == "en":
            # English detokenization using Moses detokenizer
            sentence = self._en_detok.detokenize(sentence.split())
        else:
            # Indic detokenization
            sentence = indic_detokenize.trivial_detokenize(sentence.split(), lang=language)

        return sentence.strip()


# For testing purposes:
if __name__ == "__main__":
    processor = IndicProcessor()
    sample_sentences = [
        "यह एक परीक्षण वाक्य है।",
        "This is a test sentence."
    ]

    print("Preprocessing Hindi:")
    for sent in sample_sentences:
        print(processor._preprocess(sent, "hi"))

    print("\nPreprocessing English:")
    for sent in sample_sentences:
        print(processor._preprocess(sent, "en"))

    print("\nPostprocessing Hindi:")
    for sent in sample_sentences:
        print(processor._postprocess(sent, "hi"))

    print("\nPostprocessing English:")
    for sent in sample_sentences:
        print(processor._postprocess(sent, "en"))
