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

def batch_translate(input_sentences, src_lang_ip, tgt_lang_ip, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # Preprocess using IndicProcessor with its expected language tags
        preprocessed_batch = ip.preprocess_batch(batch, src_lang_ip)
        
        # When tokenizing, we need to add the language tags that the MODEL expects
        # These are often slightly different from what IndicProcessor uses internally.
        # For ai4bharat/indictrans2, the model typically expects the format like
        # "<2en>" for English, "<2tel>" for Telugu, etc.
        # The `src_lang_ip` and `tgt_lang_ip` are for the IndicProcessor.
        # We need to map them to the model's special tokens.

        # Let's define a mapping for the special tokens
        lang_to_model_token = {
            "eng_Latn": "en", # This is what the tokenizer's _src_tokenize might expect
            "tel_Telu": "te" # This is what the tokenizer's _src_tokenize might expect
            # You might need to confirm these exact short codes from the model's documentation
            # or by inspecting the tokenizer's `special_tokens_map` or `all_special_tokens`.
            # However, for `ai4bharat/indictrans2`, the common practice is to prepend
            # target language token like "<2tel>" to the input before tokenization.
        }

        # The error suggests the tokenizer's internal `_src_tokenize` is getting `src_lang`
        # as `eng_Latn` from somewhere, and it's not in its `LANGUAGE_TAGS`.
        # The most likely place this is being passed is implicitly or explicitly
        # during the tokenization or generation process if not handled correctly.

        # Let's try specifying the target language token directly in the input to the tokenizer,
        # which is a common practice with multilingual models like mBART or NLLB.
        # The `ai4bharat/indictrans2` models are built on similar architectures.

        # Example: For English to Telugu, the input to the tokenizer should be "<2tel> English sentence"
        # The `src_lang_ip` is 'eng_Latn' and `tgt_lang_ip` is 'tel_Telu'.
        # We need to map `tgt_lang_ip` to the model's expected token, e.g., `<2tel>`.

        # Map `tgt_lang_ip` to the special token format `"<2{lang_code}>"`
        # The `IndicTrans2` models usually expect the *target* language token prepended.
        target_token_map = {
            "eng_Latn": "<2en>",
            "tel_Telu": "<2tel>",
            # Add other language mappings if your app expands
        }

        # Construct the input for the tokenizer with the target language token
        # For English to Telugu, it will be something like "<2tel> This is an English sentence."
        # For Telugu to English, it will be something like "<2en> ఇది ఒక తెలుగు వాక్యం."
        inputs_for_tokenizer = [f"{target_token_map[tgt_lang_ip]} {sentence}" for sentence in preprocessed_batch]
        
        inputs = tokenizer(inputs_for_tokenizer, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
                # Add forced_bos_token_id if specifically required, but usually
                # the target language token in the input text handles this.
                # If you still face issues, you might need to manually set this:
                # forced_bos_token_id=tokenizer.lang_code_to_id[target_token_map[tgt_lang_ip]]
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations += ip.postprocess_batch(decoded, tgt_lang_ip)
    return translations


# Load models
with st.spinner("Loading models..."):
    en_to_indic_tokenizer, en_to_indic_model = load_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
    indic_to_en_tokenizer, indic_to_en_model = load_model_and_tokenizer("ai4bharat/indictrans2-indic-en-1B")
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
            # These are the language tags for the IndicProcessor
            src_lang_ip, tgt_lang_ip = "eng_Latn", "tel_Telu"
            result = batch_translate([user_input], src_lang_ip, tgt_lang_ip, en_to_indic_model, en_to_indic_tokenizer, ip)[0]
        else:
            try:
                user_input_telugu = transliterate(user_input, ITRANS, TELUGU)
            except Exception:
                user_input_telugu = user_input

            # These are the language tags for the IndicProcessor
            src_lang_ip, tgt_lang_ip = "tel_Telu", "eng_Latn"
            result = batch_translate([user_input_telugu], src_lang_ip, tgt_lang_ip, indic_to_en_model, indic_to_en_tokenizer, ip)[0]

        st.success("Translation:")
        st.write(result)