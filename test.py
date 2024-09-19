import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from PIL import Image

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "THUDM/cogvlm2-llama3-chat-19B-int4"  # Update with the actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# Function to generate response
def generate_response(query, image=None):
    # Create a template for the conversation
    text_only_template = "A chat between a curious user and an AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

    # If no image, treat it as text-only input
    if image is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=text_only_template.format(query),
            history=[],
            template_version='chat'
        )
    else:
        # If image is provided, include it
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=[],
            images=[image],
            template_version='chat'
        )
    
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(model.device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(model.device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(model.device),
        'images': [[input_by_model['images'][0].to(model.device).to(torch.float16)]] if image is not None else None,
    }
    
    gen_kwargs = {
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Streamlit UI setup
st.title("Fitness Chatbot")

# Image upload and text input
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
query = st.text_input("Ask a question:")

# Display the uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    image = None

# Generate the response on user input
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        response = generate_response(query, image)
        st.markdown(f"**Assistant**: {response}")

# Clear conversation button
if st.button("Clear Conversation"):
    st.experimental_rerun()
