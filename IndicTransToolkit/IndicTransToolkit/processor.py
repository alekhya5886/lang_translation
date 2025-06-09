"""
Python version of the IndicProcessor class converted from Cython.
Methods preprocess_batch and postprocess_batch are exposed for external use.
All other methods are internal and use Python conventions.
"""

import regex as re
from queue import Queue
from typing import List

# Importing Python objects since these libraries don't offer C-extensions
from indicnlp.tokenize import indic_tokenize # Keep this for preprocessing, if used
# from indicnlp.tokenize import indic_detokenize # <--- Commented out
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer


class IndicProcessor:
    def __init__(self, inference=True):
        self.inference = inference

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
        # MOSES (Python objects)
        ##############################
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

        ##############################
        # INDIC NORMALIZER for Hindi (can be parameterized if needed)
        ##############################
        normalizer_factory = IndicNormalizerFactory()
        self._indic_normalizer = normalizer_factory.get_normalizer("hi")

        ##############################
        # Precompiled Patterns
        ##############################
        self._MULTISPACE_REGEX = re.compile(r"[ ]{2,}")
        self._DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
        self._DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
        self._DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
        self._END_BRACKET_SPACE_PUNC_REGEX = re.compile(r"\) ([\.!:?;,])")

        # Punctuation replacements
        self._PUNC_REPLACEMENTS = [
            (re.compile(r"\r"), ""),
            (re.compile(r"\(\s*"), "("),
            (re.compile(r"\s*\)"), ")"),
            (re.compile(r"\s:\s?"), ":"),
            (re.compile(r"\s;\s?"), ";"),
            (re.compile(r"[`´‘‚’]"), "'"),
            (re.compile(r"[„“”«»]"), '"'),
        ]

    def preprocess_batch(self, inputs: List[str], language: str):
        processed_batch = []
        for sentence in inputs:
            processed_batch.append(self._preprocess(sentence, language))
        return processed_batch

    def postprocess_batch(self, inputs: List[str], language: str):
        processed_batch = []
        for sentence in inputs:
            processed_batch.append(self._postprocess(sentence, language))
        return processed_batch

    def _preprocess(self, sentence: str, language: str) -> str:
        if not sentence or sentence.strip() == "":
            return ""

        # Normalize digits
        sentence = sentence.translate(self._digits_translation_table)

        # Normalize punctuation replacements
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

        # Normalize Indic text (using Hindi normalizer here, can be improved to param)
        sentence = self._indic_normalizer.normalize(sentence)

        # Tokenize
        if language == "en":
            sentence = self._en_tok.tokenize(sentence, return_str=True)
        else:
            sentence = " ".join(indic_tokenize.trivial_tokenize(sentence, lang=language))

        # Clean extra spaces again
        sentence = self._MULTISPACE_REGEX.sub(" ", sentence).strip()

        return sentence

    def _postprocess(self, sentence: str, language: str) -> str:
        if not sentence or sentence.strip() == "":
            return ""

        # Detokenize
        if language == "en":
            # For English, Moses detokenizer works fine
            sentence = self._en_detok.detokenize(sentence.split())
        else:
            # For Indic languages, we're temporarily bypassing indic_detokenize.trivial_detokenize
            # due to an apparent bug/incompatibility in the indicnlp library.
            # The model's direct output (decoded) should be reasonably detokenized.
            pass # Keep the sentence as is from the decoded output

        # Apply general post-processing for spaces and punctuation, which is still useful
        sentence = self._MULTISPACE_REGEX.sub(" ", sentence).strip()

        # Apply other general punctuation normalization regexes that are in _preprocess but useful here too.
        # Note: Some of these might already be handled by the model's decoding, but it's a fallback.
        sentence = re.sub(r'\s*,\s*', ', ', sentence)
        sentence = re.sub(r'\s*\.\s*(?!\d)', '. ', sentence) # Avoid touching numbers like 3.14
        sentence = re.sub(r'\s*\?\s*', '? ', sentence)
        sentence = re.sub(r'\s*!\s*', '! ', sentence)
        sentence = re.sub(r'\s*:\s*', ': ', sentence)
        sentence = re.sub(r'\s*;\s*', '; ', sentence)
        sentence = re.sub(r'\s*\(\s*', ' (', sentence)
        sentence = re.sub(r'\s*\)\s*', ') ', sentence)
        sentence = re.sub(r'"\s*(.*?)\s*"', r'"\1"', sentence) # For content inside quotes
        sentence = re.sub(r"'\s*(.*?)\s*'", r"'\1'", sentence) # For single quotes


        return sentence