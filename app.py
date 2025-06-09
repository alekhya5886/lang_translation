from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
import torch

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

@st.cache_resource
def load_model_and_tokenizer(model_path, quantization=None):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # Preprocess using IndicProcessor
        preprocessed_batch = ip.preprocess_batch(batch, src_lang)
        
        # Add source language tag as expected by the model
        # The tokenizer expects the language tag to be part of the input string
        # before the sentence, like "<lang_tag> sentence".
        # Ensure 'src_lang' here matches the format expected by the tokenizer.
        # For IndicTrans2, it's typically 'eng_Latn', 'tel_Telu', etc.
        # However, the error indicates a mismatch in the tokenizer's internal check.
        # Let's verify the correct tags from the tokenizer's configuration if possible,
        # or try the common ones.
        
        # The error specifically states "Invalid source language tag: eng_Latn"
        # within the tokenizer's internal _src_tokenize function.
        # This means that while 'eng_Latn' is used for ip.preprocess_batch,
        # the tokenizer itself might expect a different internal representation,
        # or the tag itself might not be directly part of the input to the tokenizer
        # in the same way it is for the IndicProcessor.
        #
        # Let's ensure the format for batch input to the tokenizer is correct.
        # The error points to the `tokenizer` call, not `ip.preprocess_batch`.
        # The line `batch = [f"<{src_lang}> {sentence}" for sentence in batch]`
        # is the culprit, as it's adding a tag that the tokenizer's internal
        # `_src_tokenize` doesn't recognize *when it's embedded this way*.
        #
        # The IndicTrans2 models typically expect the language tag directly during tokenization
        # or that the `processor` handles adding the necessary tokens.
        # Let's remove the manual adding of `<src_lang>` to the batch for the tokenizer.
        # The `ip.preprocess_batch` should handle the necessary language information for the model.
        
        # The `tokenizer` itself might not need `<src_lang>` prepended to the text.
        # It's possible the `IndicProcessor` takes care of adding the correct control tokens.
        # Let's try removing the line that adds the language tag to the input text.
        
        # Original problematic line:
        # batch = [f"<{src_lang}> {sentence}" for sentence in batch]
        
        # Let's feed the preprocessed batch directly to the tokenizer
        # or simply the raw batch if preprocessing already handles the internal tags.
        # Based on the error, the tokenizer itself is trying to tokenize "<eng_Latn> "
        # and doesn't consider "eng_Latn" a valid *source language tag* it handles directly
        # when embedded in the text this way.

        # Correct approach: The `IndicProcessor` handles the source/target language codes
        # for the model. The tokenizer expects the text after preprocessing.
        # The model generation usually takes care of the language direction internally based
        # on the loaded model (e.g., en-indic or indic-en).
        # We need to pass the *preprocessed* text to the tokenizer.

        # Let's re-examine the `IndicProcessor` usage with the model.
        # The `IndicTransToolkit` typically adds the language tags as special tokens
        # when it preprocesses. Let's rely on that.

        # The issue might be that `ip.preprocess_batch` returns a list of strings
        # that already have the necessary language tokens or are ready for the tokenizer.
        # Then, you're *re-adding* a tag in the format "<lang_tag> sentence",
        # which the tokenizer itself tries to parse, and it finds "eng_Latn" invalid
        # in that specific context (where it expects a different internal language identifier).

        # Solution: Pass the output of `ip.preprocess_batch` directly to the tokenizer.
        # Remove the line `batch = [f"<{src_lang}> {sentence}" for sentence in batch]`
        # after `ip.preprocess_batch`.
        
        # Use the preprocessed_batch for tokenization directly
        inputs = tokenizer(preprocessed_batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations += ip.postprocess_batch(decoded, tgt_lang)
    return translations


# Load models
with st.spinner("Loading models..."):
    # Ensure language tags used here are what the IndicProcessor expects for model initialization
    # and what the model expects for internal processing, usually the same.
    # The models themselves are "en-indic" and "indic-en".
    # The `IndicProcessor` handles the mapping of these general directions to specific language codes.
    en_to_indic_tokenizer, en_to_indic_model = load_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
    indic_to_en_tokenizer, indic_to_en_model = load_model_and_tokenizer("ai4bharat/indictrans2-indic-en-1B")
    
    # Initialize IndicProcessor once
    ip = IndicProcessor(inference=True)

# Streamlit UI
st.title("Telugu ↔ English Translator 🇮🇳")
mode = st.selectbox("Select Translation Direction", ["English ➜ Telugu", "Telugu ➜ English"])

user_input = st.text_area("Enter your sentence:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        if mode == "English ➜ Telugu":
            # For en-indic model, src_lang is "eng_Latn" and tgt_lang is "tel_Telu"
            src_lang, tgt_lang = "eng_Latn", "tel_Telu"
            result = batch_translate([user_input], src_lang, tgt_lang, en_to_indic_model, en_to_indic_tokenizer, ip)[0]
        else:
            # For indic-en model, src_lang is "tel_Telu" and tgt_lang is "eng_Latn"
            try:
                # Assuming user might type Telugu in ITRANS, convert to TELUGU script
                user_input_telugu = transliterate(user_input, ITRANS, TELUGU)
            except Exception:
                # If transliteration fails or input is already in Telugu script
                user_input_telugu = user_input

            src_lang, tgt_lang = "tel_Telu", "eng_Latn"
            result = batch_translate([user_input_telugu], src_lang, tgt_lang, indic_to_en_model, indic_to_en_tokenizer, ip)[0]

        st.success("Translation:")
        st.write(result)