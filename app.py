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

# ... (imports and load_model_and_tokenizer are similar)

def batch_translate(input_sentences, src_lang_ip, tgt_lang_ip, model, tokenizer, ip):
    translations = []
    
    model_target_tokens = {
        "eng_Latn": "<2en>",
        "tel_Telu": "<2tel>",
    }

    target_lang_token = model_target_tokens.get(tgt_lang_ip)
    
    if not target_lang_token:
        st.error(f"Error: Unsupported target language '{tgt_lang_ip}' for model token.")
        return []

    # ***CRITICAL CHANGE I SUGGESTED LAST TIME***
    tokenizer.src_lang = src_lang_ip # <--- Explicitly setting tokenizer.src_lang
    
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # 1. Preprocess using IndicProcessor (uses BCP-47 like tags)
        # KEY DIFFERENCE 1 (related): ip.preprocess_batch call
        preprocessed_batch = ip.preprocess_batch(batch, src_lang_ip) # <--- Passes only src_lang_ip to preprocess_batch
        
        # 2. Add the *target language token* to the preprocessed input.
        inputs_for_tokenizer = [f"{target_lang_token} {sentence}" for sentence in preprocessed_batch] # <--- Constructs new input with target token
        
        # 3. Tokenize the prepared input.
        # KEY DIFFERENCE 2 (related): tokenizer call
        inputs = tokenizer(inputs_for_tokenizer, truncation=True, padding="longest", return_tensors="pt").to(DEVICE) # <--- Passes 'inputs_for_tokenizer'
        
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
        
        # 4. Post-process using IndicProcessor
        # KEY DIFFERENCE 3 (related): ip.postprocess_batch call
        translations += ip.postprocess_batch(decoded, tgt_lang_ip) # <--- Passes tgt_lang_ip to postprocess_batch
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