import json
import numpy as np
import pickle
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Chatbot:
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def __init__(self, intents, model, tokenizer, max_length):
        self.intents = intents
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def start_chat(self):
        print("Chatbot is ready to talk! Type 'quit' to exit.")
        while True:
            try:
                user_response = input("You: ").strip()
                if self.make_exit(user_response):
                    print("Have a nice day!")
                    break
                response = self.generate_response(user_response)
                print(f"Bot: {response}")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nChat interrupted. Exiting...")
                break

    def make_exit(self, user_response):
        return user_response.lower() in self.exit_commands

    def generate_response(self, user_response):
        # First, check if the user_response matches any of the JSON-based responses
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                if pattern in user_response:
                    response = random.choice(intent['responses'])
                    return response

        # If no direct match, use machine learning model for prediction
        stemmed_sentence = self.get_stemmed_sentence(user_response)
        tag = self.get_predicted_tag_class(stemmed_sentence)
        reverse_dict = self.get_reverse_dict()
        tag_name = reverse_dict.get(tag, "unknown")

        print(f"Predicted Tag: {tag}")
        print(f"Tag Name: {tag_name}")

        for intent in self.intents['intents']:
            if intent['tag'] == tag_name:
                response = random.choice(intent['responses'])
                return response

        return "Sorry, I didn't understand that."

    def get_predicted_tag_class(self, stemmed_sentence):
        seq = self.tokenizer.texts_to_sequences([stemmed_sentence])
        pad_seq = pad_sequences(seq, maxlen=self.max_length, padding='post')
        x = self.model.predict(pad_seq)
        y = np.argmax(x)
        return y

    def get_reverse_dict(self):
        tags = [intent['tag'] for intent in self.intents['intents']]
        reverse_dict = {i: tag for i, tag in enumerate(tags)}
        return reverse_dict

    def get_stemmed_sentence(self, sentence):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(sentence)
        filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
        return ' '.join(filtered_tokens)


from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)


@app.route("/")
def hello():
    return "مرحبا"


@app.route("/bot", methods=["GET", "POST"])
def reply_whatsapp():
    msg = request.form.get('Body')

    chat = Chatbot(intents, model, tokenizer, max_length)
    bot_response = chat.generate_response(msg)

    response = MessagingResponse()
    response.message(bot_response)
    return str(response)


if __name__ == "__main__":
    try:
        # Load intents from JSON file
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)

        # Load machine learning model
        model = load_model("ret_chatbot.h5")

        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as infile:
            tokenizer = pickle.load(infile)

        # Load max sequence length
        with open('max_seq_length', 'rb') as infile:
            max_length = pickle.load(infile)

        # Initialize and start the chatbot
        #chat = Chatbot(intents, model, tokenizer, max_length)
        #chat.start_chat()
        app.run()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
