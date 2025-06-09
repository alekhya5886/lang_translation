from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI, TELUGU
import torch
import regex as re # For cleaning transliterated output in post-processing

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

@st.cache_resource
def load_model_and_tokenizer(model_path, quantization=None):
    """
    Loads a pre-trained Hugging Face model and tokenizer.
    Uses caching to avoid reloading on every Streamlit rerun.
    Supports 4-bit and 8-bit quantization for memory efficiency.
    """
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
        low_cpu_mem_usage=True, # Optimize for lower CPU memory usage
        quantization_config=qconfig,
    )

    if qconfig is None:
        # Move model to GPU and convert to half precision if not quantized
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval() # Set model to evaluation mode
    return tokenizer, model

def batch_translate(input_sentences: list[str], src_lang_ip: str, tgt_lang_ip: str, model, tokenizer, ip: IndicProcessor) -> list[str]:
    """
    Translates a batch of sentences using the provided model and tokenizer.
    Preprocesses input with IndicProcessor and adds language tags for the model.
    """
    translations = []
    
    # Iterate through input sentences in batches
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # 1. Preprocess batch using IndicProcessor.
        # This handles normalization and other language-specific preprocessing for the source.
        preprocessed_batch = ip.preprocess_batch(batch, src_lang_ip) 
        
        # 2. Construct input for the IndicTrans2 tokenizer.
        # The tokenizer expects the format "src_lang_code tgt_lang_code actual_text".
        inputs_for_tokenizer = [f"{src_lang_ip} {tgt_lang_ip} {sentence}" for sentence in preprocessed_batch]
        
        # 3. Tokenize the prepared input.
        # This converts the text into numerical IDs that the model can understand.
        inputs = tokenizer(
            inputs_for_tokenizer, 
            truncation=True, # Truncate long sentences
            padding="longest", # Pad shorter sentences to the longest in the batch
            return_tensors="pt" # Return PyTorch tensors
        ).to(DEVICE)

        # 4. Generate translations using the model.
        with torch.no_grad(): # Disable gradient calculation for inference (saves memory and speeds up)
            generated_tokens = model.generate(
                **inputs,
                use_cache=True, # Use cache for faster generation
                min_length=0, # Minimum length of generated sequence
                max_length=256, # Maximum length of generated sequence
                num_beams=5, # Number of beams for beam search (for better quality)
                num_return_sequences=1, # Return only the top sequence
            )

        # 5. Decode the generated token IDs back into human-readable text.
        # `skip_special_tokens=True` removes special tokens like [CLS], [SEP], language tags.
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Add decoded translations to the list.
        # For this setup, we are directly using the decoded output, and applying
        # post-processing (like script conversion) later in the Streamlit UI logic.
        translations.extend(decoded) 

    return translations

# --- Main Streamlit Application Logic ---

# Load models
with st.spinner("Loading models..."):
    # Load English to Indic (e.g., English to Telugu) model
    en_to_indic_tokenizer, en_to_indic_model = load_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
    
    # Load Indic to English (e.g., Telugu to English) model
    indic_to_en_tokenizer, indic_to_en_model = load_model_and_tokenizer("ai4bharat/indictrans2-indic-en-1B")
    
    # Initialize IndicProcessor for text normalization and preparation
    ip = IndicProcessor(inference=True)

# Streamlit UI
st.title("Telugu ↔ English Translator 🇮🇳")
mode = st.selectbox("Select Translation Direction", ["English ➜ Telugu", "Telugu ➜ English"])

user_input = st.text_area("Enter your sentence:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        result = "" # Initialize result variable
        if mode == "English ➜ Telugu":
            src_lang_ip, tgt_lang_ip = "eng_Latn", "tel_Telu"
            
            # Get raw translation from the model (might be in Devanagari even for Telugu target)
            raw_translation_list = batch_translate([user_input], src_lang_ip, tgt_lang_ip, en_to_indic_model, en_to_indic_tokenizer, ip)
            raw_translation = raw_translation_list[0]
            
            # Post-process: Transliterate from Devanagari to Telugu script if necessary.
            # This handles cases where the model outputs Telugu words in Devanagari script.
            try:
                result = transliterate(raw_translation, DEVANAGARI, TELUGU)
            except Exception as e:
                # Fallback to raw if transliteration fails (e.g., input wasn't Devanagari)
                st.info(f"Could not transliterate from Devanagari to Telugu for Eng->Tel. Raw output used: {e}")
                result = raw_translation 

        else: # Telugu ➜ English
            src_lang_ip, tgt_lang_ip = "tel_Telu", "eng_Latn"
            
            # For Telugu to English, we assume user input is already in Telugu script.
            # No ITRANS transliteration of input is needed here.
            user_input_for_model = user_input 
            
            # Get raw translation from the model.
            # As confirmed, this model outputs in Devanagari for Telugu->English.
            raw_translation_list = batch_translate([user_input_for_model], src_lang_ip, tgt_lang_ip, indic_to_en_model, indic_to_en_tokenizer, ip)
            raw_translation = raw_translation_list[0]

            # --- MANDATORY POST-PROCESSING FOR TELUGU->ENGLISH ---
            # This step is critical because the IndicTrans2-Indic-En model
            # typically outputs in Devanagari script for Telugu->English,
            # not directly in Latin script. We force it to Latin script (ITRANS).
            translated_text_latin = raw_translation # Initialize with raw output

            try:
                # 1. Transliterate from Devanagari (model's actual output) to ITRANS (Latin-based scheme)
                # This ensures the output is always in Latin characters.
                translated_text_latin = transliterate(raw_translation, DEVANAGARI, ITRANS)
                
                # 2. Apply common heuristics to make ITRANS output more readable as English.
                # These are approximations and will not fix semantic translation errors.
                translated_text_latin = translated_text_latin.replace("aa", "a").replace("ii", "i").replace("uu", "u")
                translated_text_latin = translated_text_latin.replace("ee", "e").replace("oo", "o")
                translated_text_latin = translated_text_latin.replace("ai", "e").replace("au", "o") 
                
                # Remove any remaining non-standard Latin characters or transliteration marks.
                translated_text_latin = re.sub(r'[\u0300-\u036F]', '', translated_text_latin) # Combining diacritics
                translated_text_latin = translated_text_latin.replace('~', '').replace('\'', '') # Tilde, apostrophe
                translated_text_latin = translated_text_latin.replace('`', '').replace('|', '').replace('.', '') # Backtick, danda, period if not sentence end
                
                # Clean up multiple spaces that might result from replacements
                translated_text_latin = re.sub(r'\s+', ' ', translated_text_latin).strip()

                # Capitalize the first letter of the sentence for readability
                if translated_text_latin:
                    result = translated_text_latin[0].upper() + translated_text_latin[1:]
                else:
                    result = "" # Handle empty string case

            except Exception as e:
                # If transliteration fails (e.g., input was already Latin, or unusual characters)
                st.warning(f"Error during Devanagari to English script transliteration (Tel->Eng): {e}. Displaying raw model output (which might be in Devanagari or problematic Latin).")
                result = raw_translation # Fallback to raw output if script conversion fails

        st.success("Translation:")
        st.write(result)