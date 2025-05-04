import nltk
import random
import string
import speech_recognition as sr
import pyttsx3
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class SmartChatBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Initialize knowledge base
        self.knowledge_base = {
            "greetings": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to ask if you need anything else!"
            ],
            "thanks": [
                "You're welcome!",
                "Happy to help!",
                "My pleasure!"
            ],
            "unknown": [
                "I'm not sure I understand. Could you rephrase that?",
                "I'm still learning. Could you explain differently?",
                "I don't have information about that yet."
            ]
        }
        
        # Task-specific responses
        self.tasks = {
            "time": self.get_time,
            "weather": self.get_weather,
            "calculate": self.calculate,
            "search": self.web_search
        }

    def preprocess_text(self, text):
        """Preprocess input text for better understanding"""
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in string.punctuation and token not in stop_words]
        
        return tokens

    def listen(self):
        """Listen to user's voice input"""
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                self.speak("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError:
                self.speak("Sorry, there was an error with the speech service.")
                return None

    def speak(self, response):
        """Convert text response to speech"""
        print(f"Bot: {response}")
        self.engine.say(response)
        self.engine.runAndWait()

    def get_response(self, user_input):
        """Generate appropriate response based on user input"""
        if not user_input:
            return random.choice(self.knowledge_base["unknown"])

        tokens = self.preprocess_text(user_input)
        
        # Check for greetings
        greeting_words = {"hello", "hi", "hey", "greetings"}
        if any(word in tokens for word in greeting_words):
            return random.choice(self.knowledge_base["greetings"])
        
        # Check for farewell
        farewell_words = {"bye", "goodbye", "cya", "farewell"}
        if any(word in tokens for word in farewell_words):
            return random.choice(self.knowledge_base["farewell"])
        
        # Check for thanks
        thanks_words = {"thanks", "thank", "appreciate"}
        if any(word in tokens for word in thanks_words):
            return random.choice(self.knowledge_base["thanks"])
        
        # Check for specific tasks
        for task_keyword, task_function in self.tasks.items():
            if task_keyword in tokens:
                return task_function(user_input)
        
        return self.generate_response(user_input)

    def generate_response(self, user_input):
        """Generate response using TF-IDF and cosine similarity"""
        # This is a simple implementation - you can expand this
        responses = [
            "I understand you're asking about {}",
            "Let me help you with {}",
            "Here's what I know about {}"
        ]
        
        # Extract key terms
        tokens = self.preprocess_text(user_input)
        if tokens:
            key_term = tokens[0]
            return random.choice(responses).format(key_term)
        
        return random.choice(self.knowledge_base["unknown"])

    # Task-specific methods
    def get_time(self, _):
        """Get current time"""
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def get_weather(self, _):
        """Placeholder for weather information"""
        return "I'm sorry, I don't have access to weather information yet."

    def calculate(self, expression):
        """Simple calculator function"""
        try:
            # Extract numbers and operation
            tokens = self.preprocess_text(expression)
            numbers = []
            operation = None
            for token in tokens:
                if token.isdigit():
                    numbers.append(int(token))
                elif token in ['plus', 'add', 'minus', 'subtract', 'multiply', 'divide']:
                    operation = token
            
            if len(numbers) == 2 and operation:
                if operation in ['plus', 'add']:
                    return f"The result is {numbers[0] + numbers[1]}"
                elif operation in ['minus', 'subtract']:
                    return f"The result is {numbers[0] - numbers[1]}"
                elif operation == 'multiply':
                    return f"The result is {numbers[0] * numbers[1]}"
                elif operation == 'divide':
                    return f"The result is {numbers[0] / numbers[1]}"
        except:
            pass
        return "I couldn't perform that calculation."

    def web_search(self, query):
        """Placeholder for web search functionality"""
        return "I'm sorry, I don't have web search capabilities yet."

def main():
    chatbot = SmartChatBot()
    print("Bot: Hello! I'm your voice-enabled assistant. How can I help you?")
    
    while True:
        # Get voice input
        user_input = chatbot.listen()
        
        if user_input:
            # Generate and speak response
            response = chatbot.get_response(user_input)
            chatbot.speak(response)
            
            # Check for exit command
            if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
                break

if __name__ == "__main__":
    main() 