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
    
    # Define the special tokens for the model
    # These are the target language tokens that the IndicTrans2 models expect
    # to be prepended to the input for translation direction.
    # Note: These are different from the 'src_lang' argument in the _src_tokenize
    # function that is causing the error. The error is about the *source* language
    # as perceived by the tokenizer's internal mechanisms, not the target token
    # we prepend.
    
    # The `IndicTransToolkit` handles the `src_lang` and `tgt_lang` internally
    # when you use `ip.preprocess_batch` and `ip.postprocess_batch`.
    # For the `tokenizer` and `model.generate` call, we need to ensure the input
    # format is correct, which often includes the *target* language token.
    
    # Let's map the IndicProcessor's target language codes to the model's special tokens
    model_target_tokens = {
        "eng_Latn": "<2en>",
        "tel_Telu": "<2tel>",
        # Add other languages as needed
    }

    # Get the target language token for the current translation
    target_lang_token = model_target_tokens.get(tgt_lang_ip)
    
    if not target_lang_token:
        st.error(f"Error: Unsupported target language '{tgt_lang_ip}' for model token.")
        return []

    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # 1. Preprocess using IndicProcessor (uses BCP-47 like tags)
        # This handles normalization and other language-specific preprocessing.
        preprocessed_batch = ip.preprocess_batch(batch, src_lang_ip)
        
        # 2. Add the *target language token* to the preprocessed input.
        # This tells the model which language to translate *into*.
        # The tokenizer is part of the model's pipeline, and it expects this.
        inputs_for_tokenizer = [f"{target_lang_token} {sentence}" for sentence in preprocessed_batch]
        
        # 3. Tokenize the prepared input.
        # The key insight here is that `_src_tokenize` error likely occurs when the
        # tokenizer implicitly tries to derive the source language from the input
        # or its configuration, and it doesn't recognize the value it gets.
        # By providing the target language token correctly, we are guiding the model.
        # Also, ensure no *other* source language parameter is being passed that conflicts.
        inputs = tokenizer(inputs_for_tokenizer, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
                # It's crucial NOT to pass a `src_lang` parameter directly here
                # if the tokenizer isn't expecting it or has a conflict.
                # The model often infers the source from the input without an explicit src_lang arg.
                # If you absolutely *must* set `forced_bos_token_id`, ensure it's
                # the ID of the *target* language token, not a source language tag.
                # Example: forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_token.strip('<>')]
                # However, usually, prepending the token to the input is sufficient.
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 4. Post-process using IndicProcessor
        translations += ip.postprocess_batch(decoded, tgt_lang_ip)
    return translations


# Load models
with st.spinner("Loading models..."):
    # Load the English-to-Indic model and tokenizer
    en_to_indic_tokenizer, en_to_indic_model = load_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
    
    # Load the Indic-to-English model and tokenizer
    indic_to_en_tokenizer, indic_to_en_model = load_model_and_tokenizer("ai4bharat/indictrans2-indic-en-1B")
    
    # Initialize IndicProcessor
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
            # Languages for IndicProcessor's preprocess/postprocess methods
            src_lang_ip, tgt_lang_ip = "eng_Latn", "tel_Telu"
            result = batch_translate([user_input], src_lang_ip, tgt_lang_ip, en_to_indic_model, en_to_indic_tokenizer, ip)[0]
        else:
            # For Telugu to English
            try:
                # Attempt to transliterate if the input is in ITRANS (Latin script)
                user_input_telugu = transliterate(user_input, ITRANS, TELUGU)
            except Exception:
                # If transliteration fails (e.g., already in Telugu script or invalid ITRANS)
                user_input_telugu = user_input

            # Languages for IndicProcessor's preprocess/postprocess methods
            src_lang_ip, tgt_lang_ip = "tel_Telu", "eng_Latn"
            result = batch_translate([user_input_telugu], src_lang_ip, tgt_lang_ip, indic_to_en_model, indic_to_en_tokenizer, ip)[0]

        st.success("Translation:")
        st.write(result)