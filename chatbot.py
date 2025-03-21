import streamlit as st
import streamlit_authenticator as stauth
import os
import json
import PyPDF2
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import Dataset

# Define user credentials with hashed passwords
credentials = {
    'usernames': {
        'teacher1': {
            'name': 'Daniel',
            'password': '$2b$12$KIX1p1z5f7g6s9s8d9s9sO'  # Replace with actual hashed password
        },
        'student1': {
            'name': 'Student One',
            'password': '$2b$12$KIX1p1z5f7g6s9s8d9s9sO'  # Replace with actual hashed password
        }
        # Add more users as needed
    }
}

# Initialize the authenticator
authenticator = stauth.Authenticate(
    credentials,
    'cookie_name',
    'signature_key',
    cookie_expiry_days=30
)

# Render the login widget
name, authentication_status = authenticator.login(location= 'main')

if authentication_status:
    st.write(f'Welcome *{name}*')

    # Function to load chat history
    def load_chat_history(username):
        if os.path.exists(f'{username}_chat_history.json'):
            with open(f'{username}_chat_history.json', 'r') as file:
                return json.load(file)
        return []

    # Function to save chat history
    def save_chat_history(username, chat_history):
        with open(f'{username}_chat_history.json', 'w') as file:
            json.dump(chat_history, file)

    # Load user's chat history
    chat_history = load_chat_history(username)

    # Display chat history
    st.write("### Chat History")
    for message in chat_history:
        st.write(f"{message['role']}: {message['content']}")

    # Input for new message
    user_input = st.text_input("You: ")

    # Check if the fine-tuned model exists
    model_path = "./fine_tuned_model"
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        st.warning("Fine-tuned model not found. Please upload a PDF and fine-tune the model.")

    if user_input:
        if 'text_generator' in locals():
            # Generate a response using the fine-tuned model
            response = text_generator(
                user_input,
                max_length=100,  # Increase max length for longer responses
                num_return_sequences=1,
                temperature=0.7,  # Adjust temperature for creativity
                top_k=50,  # Limit to top-k tokens
                top_p=0.9  # Use nucleus sampling
            )[0]['generated_text']
        else:
            # Placeholder response if the model is not fine-tuned
            response = "This is a response from the model."

        # Append new message to chat history
        chat_history.append({'role': 'user', 'content': user_input})
        chat_history.append({'role': 'bot', 'content': response})
        # Save updated chat history
        save_chat_history(username, chat_history)
        # Display the new message and response
        st.write(f"You: {user_input}")
        st.write(f"Bot: {response}")

    # Upload PDF
    st.write("### Upload a PDF for Fine-Tuning")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Function to extract text from PDF
    def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Skip empty pages
                text += page_text + "\n"  # Add a newline between pages
        return text.strip()  # Remove leading/trailing whitespace

    # Function to clean text
    def clean_text(text):
        # Remove special characters and numbers
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    # Fine-tune the model
    if uploaded_file is not None:
        if st.button("Fine-Tune Model"):
            with st.spinner("Fine-tuning the model..."):
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_file)
                # Clean the extracted text
                pdf_text = clean_text(pdf_text)

                # Save extracted text to a file
                with open("pdf_text.txt", "w") as file:
                    file.write(pdf_text)

                # Load tokenizer and model
                model_name = "gpt2"  # Replace with desired model name
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)

                # Prepare the dataset
                dataset = Dataset.from_dict({"text": [pdf_text]})

                # Tokenize the dataset
                def tokenize_function(examples):
                    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

                tokenized_dataset = dataset.map(tokenize_function, batched=True)

                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir="./fine_tuned_model",
                    overwrite_output_dir=True,
                    num_train_epochs=3,  # Train for more epochs
                    per_device_train_batch_size=2,  # Reduce batch size for larger models
                    save_steps=10_000,
                    save_total_limit=2,
                    logging_dir="./logs",  # Log training progress
                    logging_steps=500,
                )

                # Initialize the Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                )

                # Fine-tune the model
                trainer.train()

                # Save the fine-tuned model
                model.save_pretrained("./fine_tuned_model")
                tokenizer.save_pretrained("./fine_tuned_model")

                st.success("Model fine-tuned with the uploaded PDF content!")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
