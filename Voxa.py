import nltk
import random
import string
import json
import os
import requests
import speech_recognition as sr
import pyttsx3
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

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

        # Weather API configuration
        self.weather_api_key = "292c06d6befbb0f88367adefb2bbca2e"  # Replace with your actual API key

        # Google Search API configuration
        self.search_api_key = "AIzaSyA_zItcvgikJqOQZWJ2BIPqlh210sxx6vc"  # Replace with your actual API key
        self.search_engine_id = "d3845d08da67e43d0"  # Replace with your actual Search Engine ID

        # Flag to determine if APIs are available
        self.apis_available = self.check_api_availability()

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

        # Load extended knowledge base if exists
        self.extended_kb_file = "knowledge_base.json"
        self.extended_knowledge = self.load_knowledge_base()

        # Conversation context
        self.conversation_history = []
        self.max_history = 5

        # Task-specific responses
        self.tasks = {
            "time": self.get_time,
            "weather": self.get_weather,
            "calculate": self.calculate,
            "search": self.web_search
        }

    def check_api_availability(self):
        """Check if the configured APIs are available and valid"""
        apis_ok = True

        # Check if Weather API key is valid
        if self.weather_api_key == "YOUR_OPENWEATHERMAP_API_KEY" or not self.weather_api_key:
            print("Warning: OpenWeatherMap API key not configured. Weather functionality will be limited.")
            apis_ok = False

        # Check if Google Search API key is valid
        if self.search_api_key == "YOUR_GOOGLE_API_KEY" or not self.search_api_key:
            print("Warning: Google API key not configured. Web search functionality will be limited.")
            apis_ok = False

        # Check if Search Engine ID is valid
        if self.search_engine_id == "YOUR_SEARCH_ENGINE_ID" or not self.search_engine_id:
            print("Warning: Search Engine ID not configured. Web search functionality will be limited.")
            apis_ok = False

        return apis_ok

    def load_knowledge_base(self):
        """Load extended knowledge base from JSON file"""
        if os.path.exists(self.extended_kb_file):
            try:
                with open(self.extended_kb_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")

        # Create default knowledge base if file doesn't exist
        default_kb = {
            "facts": {
                "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
                "chatbot": "A chatbot is a software application that uses artificial intelligence to simulate conversation with users.",
                "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
                "calculation": "I can perform various calculations. Try asking me something like 'calculate 5 plus 3' or 'what is 10 times 4'.",
                "weather": "I can provide weather information for any location. Try asking 'what's the weather in London' or 'weather forecast for New York'.",
                "search": "I can search the web for information. Try saying 'search for latest news' or 'find information about climate change'."
            },
            "qa_pairs": [
                {"question": "What is your name?", "answer": "I am a voice-enabled smart chatbot designed to assist you."},
                {"question": "How do you work?", "answer": "I use natural language processing to understand your queries and provide relevant responses."},
                {"question": "Who created you?", "answer": "I was created as a Python programming project to demonstrate NLP and voice interaction capabilities."},
                {"question": "Can you do calculations?", "answer": "Yes, I can perform basic calculations. Try asking me something like 'calculate 5 plus 3' or 'what is 10 times 4'."},
                {"question": "Can you check the weather?", "answer": "Yes, I can provide weather information for any location. Try asking 'what's the weather in London' or 'weather forecast for New York'."},
                {"question": "Can you search the web?", "answer": "Yes, I can search the web for information. Try saying 'search for latest news' or 'find information about climate change'."}
            ]
        }

        # Save default knowledge base
        with open(self.extended_kb_file, 'w') as f:
            json.dump(default_kb, f, indent=4)

        return default_kb

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
        """Listen to user's voice input with text fallback"""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    print("Speech not recognized. Please type your message instead:")
                    return self.text_input_fallback()
                except sr.RequestError:
                    print("Speech service error. Please type your message instead:")
                    return self.text_input_fallback()
        except Exception as e:
            print(f"Microphone error: {e}")
            print("Using text input mode instead:")
            return self.text_input_fallback()

    def text_input_fallback(self):
        """Fallback to text input when speech recognition fails"""
        try:
            text = input("You: ")
            return text
        except Exception as e:
            print(f"Input error: {e}")
            return "help"  # Default fallback command

    def speak(self, response):
        """Convert text response to speech"""
        print(f"Bot: {response}")
        self.engine.say(response)
        self.engine.runAndWait()

    def get_response(self, user_input):
        """Generate appropriate response based on user input"""
        if not user_input:
            return random.choice(self.knowledge_base["unknown"])

        # Add to conversation history
        self.conversation_history.append({"user": user_input})
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        # Check for help command
        if user_input.lower() in ["help", "commands", "what can you do", "capabilities"]:
            return self.get_help_info()

        tokens = self.preprocess_text(user_input)

        # Analyze sentiment
        sentiment = self.analyze_sentiment(user_input)

        # Check for greetings
        greeting_words = {"hello", "hi", "hey", "greetings"}
        if any(word in tokens for word in greeting_words):
            response = random.choice(self.knowledge_base["greetings"])
            self.conversation_history.append({"bot": response})
            return response

        # Check for farewell
        farewell_words = {"bye", "goodbye", "cya", "farewell"}
        if any(word in tokens for word in farewell_words):
            response = random.choice(self.knowledge_base["farewell"])
            self.conversation_history.append({"bot": response})
            return response

        # Check for thanks
        thanks_words = {"thanks", "thank", "appreciate"}
        if any(word in tokens for word in thanks_words):
            response = random.choice(self.knowledge_base["thanks"])
            self.conversation_history.append({"bot": response})
            return response

        # Check for specific tasks using more flexible matching
        task_indicators = {
            "time": ["time", "clock", "hour", "what time", "current time", "tell me the time"],
            "weather": ["weather", "temperature", "forecast", "climate", "how hot", "how cold", "raining", "sunny"],
            "calculate": ["calculate", "compute", "math", "add", "subtract", "plus", "minus", "multiply", "divide", "sum", "difference", "equals", "what is", "what's", "solve"],
            "search": ["search", "find", "look up", "google", "information about", "info on", "tell me about", "what is", "who is"]
        }

        # Check each task category
        for task_name, indicators in task_indicators.items():
            # Check if any indicator is in the original input (case insensitive)
            if any(indicator.lower() in user_input.lower() for indicator in indicators):
                task_function = self.tasks.get(task_name)
                if task_function:
                    response = task_function(user_input)
                    self.conversation_history.append({"bot": response})
                    return response

        # Check knowledge base for direct matches
        response = self.query_knowledge_base(user_input)
        if response:
            self.conversation_history.append({"bot": response})
            return response

        # Generate response based on context and input
        response = self.generate_response(user_input, sentiment)
        self.conversation_history.append({"bot": response})
        return response

    def get_help_info(self):
        """Provide information about the chatbot's capabilities"""
        # Check which APIs are available
        weather_status = "✓ Available" if self.apis_available and self.weather_api_key != "YOUR_OPENWEATHERMAP_API_KEY" else "✗ Not configured"
        search_status = "✓ Available" if self.apis_available and self.search_api_key != "YOUR_GOOGLE_API_KEY" and self.search_engine_id != "YOUR_SEARCH_ENGINE_ID" else "✗ Not configured"

        help_text = f"""
I can help you with the following:

1. Answer questions about various topics from my knowledge base
2. Provide weather information ({weather_status})
   Examples: "What's the weather in London?", "Temperature in New York", "Weather forecast for Tokyo"
3. Perform web searches ({search_status})
   Examples: "Search for latest AI developments", "Find information about climate change", "Look up Python programming"
4. Tell you the current time
   Examples: "What time is it?", "Tell me the current time", "What's the time now?"
5. Perform calculations
   Examples: "Calculate 5 plus 3", "What is 10 times 4?", "Compute 25 divided by 5", "What's 7 minus 2?"

You can phrase your questions naturally - I'll do my best to understand what you're asking!

Voice and Text Commands:
- Say or type 'text mode' to switch to text-only input
- Say or type 'voice mode' to switch back to voice input
- Say or type 'help' to see this information again
- Say or type 'goodbye' or 'exit' to end our conversation

Tips for best results:
- Speak clearly and at a moderate pace
- For calculations, clearly state the numbers and operation
- For weather, include the location name
- If I don't understand, try rephrasing your question
"""
        return help_text

    def analyze_sentiment(self, text):
        """Analyze sentiment of input text"""
        try:
            analysis = TextBlob(text)
            # Returns polarity between -1 (negative) and 1 (positive)
            return analysis.sentiment.polarity
        except:
            return 0  # Neutral sentiment as fallback

    def query_knowledge_base(self, user_input):
        """Query the extended knowledge base for relevant information"""
        # Check for direct fact matches
        tokens = self.preprocess_text(user_input)
        for key, fact in self.extended_knowledge.get("facts", {}).items():
            if key in tokens:
                return fact

        # Check for QA pairs
        for qa_pair in self.extended_knowledge.get("qa_pairs", []):
            question = qa_pair.get("question", "").lower()
            if self.calculate_similarity(user_input.lower(), question) > 0.7:
                return qa_pair.get("answer")

        return None

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using TF-IDF and cosine similarity"""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0

    def get_conversation_context(self):
        """Extract context from conversation history"""
        if len(self.conversation_history) < 2:
            return ""

        # Get the last few exchanges
        recent_history = self.conversation_history[-4:]
        context = ""
        for exchange in recent_history:
            if "user" in exchange:
                context += f"User said: {exchange['user']} "
            elif "bot" in exchange:
                context += f"Bot responded: {exchange['bot']} "

        return context

    def generate_response(self, user_input, sentiment=0):
        """Generate response using advanced NLP techniques"""
        # Get conversation context
        context = self.get_conversation_context()
        context_aware = len(context) > 0

        # Extract key terms for response generation
        tokens = self.preprocess_text(user_input)

        # Adjust response based on sentiment
        if sentiment > 0.5:
            sentiment_prefix = "I'm glad to hear that! "
        elif sentiment < -0.5:
            sentiment_prefix = "I'm sorry to hear that. "
        else:
            sentiment_prefix = ""

        # Generate response based on key terms and context
        if tokens:
            # Try to find the most relevant key term
            key_terms = [token for token in tokens if len(token) > 3]
            if key_terms:
                key_term = key_terms[0]

                # Check if we have any facts about this term
                for term, fact in self.extended_knowledge.get("facts", {}).items():
                    if self.calculate_similarity(key_term, term) > 0.7:
                        # Add context awareness if we have conversation history
                        if context_aware:
                            return f"{sentiment_prefix}Based on our conversation, here's information about {term}: {fact}"
                        return sentiment_prefix + fact

                # Default response with key term
                if context_aware:
                    # Context-aware responses
                    responses = [
                        "Continuing our conversation, I understand you're asking about {}.",
                        "Based on what we've discussed, let me help you with {}.",
                        "Considering our previous exchange, here's what I know about {}."
                    ]
                else:
                    # Standard responses
                    responses = [
                        "I understand you're asking about {}.",
                        "Let me help you with {}.",
                        "Here's what I know about {}."
                    ]
                return sentiment_prefix + random.choice(responses).format(key_term)

        # If we have context but couldn't generate a specific response
        if context_aware:
            return sentiment_prefix + "I'm not sure I understand. Could you provide more details about what you're asking?"

        return sentiment_prefix + random.choice(self.knowledge_base["unknown"])

    # Task-specific methods
    def get_time(self, _):
        """Get current time"""
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def get_weather(self, query):
        """Get weather information for a location"""
        # Check if API is available
        if not self.apis_available or self.weather_api_key == "YOUR_OPENWEATHERMAP_API_KEY":
            return "I'm sorry, but my weather service is not configured. I can help you with calculations, knowledge base queries, or other tasks instead."

        try:
            # Extract location from query
            location = self.extract_location(query)
            if not location:
                return "I need a location to check the weather. Please specify a city or place like 'weather in London' or 'what's the temperature in New York'."

            # Call OpenWeatherMap API
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                weather_desc = data['weather'][0]['description']
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed']

                return (f"Weather in {location}: {weather_desc}. "
                        f"Temperature: {temp}°C. "
                        f"Humidity: {humidity}%. "
                        f"Wind speed: {wind_speed} m/s.")
            elif response.status_code == 401:
                print("Weather API error: Invalid API key")
                return "I'm sorry, but my weather service is not properly configured. I can help you with calculations, knowledge base queries, or other tasks instead."
            else:
                return f"Sorry, I couldn't find weather information for {location}. Please try another location or check your spelling."
        except Exception as e:
            print(f"Weather API error: {e}")
            return "I'm having trouble getting weather information right now. Let me help you with something else instead. You can ask me to calculate something or search my knowledge base."

    def extract_location(self, query):
        """Extract location from user query using improved methods"""
        import re

        # Try to find location patterns like "in New York" or "for London"
        location_indicators = ["in", "for", "at", "of", "near", "around"]

        # First, check for common patterns with location indicators
        for indicator in location_indicators:
            pattern = rf'\b{indicator}\s+([A-Za-z\s]+)(?:\b|$)'
            match = re.search(pattern, query)
            if match:
                location = match.group(1).strip()
                # If it's multiple words, capitalize each word
                return ' '.join(word.capitalize() for word in location.split())

        # Next, try to find capitalized words that might be locations
        capitalized_words = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
        if capitalized_words:
            return capitalized_words[0]

        # If no capitalized words, look for common city names in the query
        common_cities = ["new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
                        "san antonio", "san diego", "dallas", "san jose", "austin", "london", "paris",
                        "tokyo", "delhi", "shanghai", "sao paulo", "mexico city", "cairo", "mumbai",
                        "beijing", "dhaka", "osaka", "karachi", "istanbul", "moscow", "bangalore"]

        query_lower = query.lower()
        for city in common_cities:
            if city in query_lower:
                return ' '.join(word.capitalize() for word in city.split())

        # If all else fails, extract any word that might be a location
        words = query.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ["what", "where", "when", "how", "tell", "give", "show", "find", "weather", "temperature", "forecast", "climate"]:
                return word.capitalize()

        return None

    def calculate(self, expression):
        """Advanced calculator function"""
        try:
            # First try to extract a mathematical expression using regex
            import re

            # Extract numbers from the original expression (before preprocessing)
            numbers = re.findall(r'\d+\.?\d*', expression)
            numbers = [float(num) if '.' in num else int(num) for num in numbers]

            # Look for operation keywords in the original expression
            operation_map = {
                'plus': '+', 'add': '+', 'sum': '+', 'addition': '+',
                'minus': '-', 'subtract': '-', 'subtraction': '-', 'difference': '-',
                'times': '*', 'multiply': '*', 'multiplication': '*', 'product': '*',
                'divide': '/', 'division': '/', 'divided by': '/'
            }

            operation = None
            for op_word, op_symbol in operation_map.items():
                if op_word in expression.lower():
                    operation = op_symbol
                    break

            # If we have two numbers and an operation, perform the calculation
            if len(numbers) == 2 and operation:
                result = None
                if operation == '+':
                    result = numbers[0] + numbers[1]
                elif operation == '-':
                    result = numbers[0] - numbers[1]
                elif operation == '*':
                    result = numbers[0] * numbers[1]
                elif operation == '/':
                    if numbers[1] != 0:  # Avoid division by zero
                        result = numbers[0] / numbers[1]
                    else:
                        return "I can't divide by zero."

                if result is not None:
                    # Format the result nicely (remove trailing zeros for floats)
                    if isinstance(result, float) and result.is_integer():
                        result = int(result)
                    return f"The result of {numbers[0]} {operation} {numbers[1]} is {result}."

            # If the above method fails, try direct evaluation (for expressions like "5+3")
            # First, extract a potential math expression using regex
            math_expr = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', expression)
            if math_expr:
                # Safely evaluate the expression
                result = eval(math_expr.group(1))
                return f"The result is {result}."

            # If all else fails, try to extract a direct calculation request
            if "calculate" in expression.lower() or "compute" in expression.lower() or "what is" in expression.lower():
                # Try to find a mathematical expression in the query
                expr = expression.lower()
                for word in ["calculate", "compute", "what is", "what's", "solve"]:
                    if word in expr:
                        expr = expr.split(word, 1)[1].strip()
                        break

                # Try to evaluate the expression safely
                try:
                    # Replace words with symbols
                    for word, symbol in [('plus', '+'), ('minus', '-'), ('times', '*'), ('divided by', '/'), ('multiply by', '*')]:
                        expr = expr.replace(word, symbol)

                    # Clean up the expression
                    expr = re.sub(r'[^0-9+\-*/().]', '', expr)

                    if expr:
                        result = eval(expr)
                        return f"The result is {result}."
                except:
                    pass
        except Exception as e:
            print(f"Calculation error: {e}")

        return "I couldn't perform that calculation. Please try phrasing it differently, like '5 plus 3' or 'calculate 10 minus 2'."

    def web_search(self, query):
        """Perform web search using Google Custom Search API"""
        # Check if API is available
        if not self.apis_available or self.search_api_key == "YOUR_GOOGLE_API_KEY" or self.search_engine_id == "YOUR_SEARCH_ENGINE_ID":
            return "I'm sorry, but my web search service is not configured. I can help you with calculations, knowledge base queries, or other tasks instead."

        try:
            # Extract search query
            search_query = self.extract_search_query(query)
            if not search_query:
                return "What would you like me to search for? Please provide more details."

            # Create a service object for the Google Custom Search API
            service = build("customsearch", "v1", developerKey=self.search_api_key)

            # Execute the search
            result = service.cse().list(q=search_query, cx=self.search_engine_id, num=3).execute()

            # Process results
            if "items" in result:
                response = f"Here are some results for '{search_query}':\n\n"
                for i, item in enumerate(result["items"][:3], 1):
                    response += f"{i}. {item['title']}\n"
                    response += f"   {item['snippet']}\n"
                    response += f"   URL: {item['link']}\n\n"
                return response
            else:
                # Try to provide some information from the knowledge base instead
                for key, fact in self.extended_knowledge.get("facts", {}).items():
                    if key in search_query.lower():
                        return f"I couldn't find web results, but I know that {fact}"

                return f"I couldn't find any results for '{search_query}'. Please try a different search term or ask me something else."
        except Exception as e:
            print(f"Web search error: {e}")
            error_msg = str(e).lower()

            if "invalid credentials" in error_msg or "api key" in error_msg:
                return "I'm sorry, but my web search service is not properly configured. I can help you with calculations, knowledge base queries, or other tasks instead."

            return "I'm having trouble performing the web search right now. Let me help you with something else instead. You can ask me to calculate something or search my knowledge base."

    def extract_search_query(self, query):
        """Extract search query from user input with improved methods"""
        import re

        # List of phrases that indicate a search request
        search_indicators = [
            "search for", "search about", "search",
            "find information on", "find info on", "find information about", "find info about", "find out about", "find",
            "look up", "lookup", "google",
            "information about", "info on", "info about",
            "tell me about", "what is", "who is", "where is", "when is", "why is", "how is",
            "what are", "who are", "where are", "when are", "why are", "how are"
        ]

        clean_query = query.lower()

        # Try to extract the search query using the indicators
        for indicator in search_indicators:
            if indicator in clean_query:
                # Get everything after the indicator
                search_query = clean_query.split(indicator, 1)[1].strip()
                if search_query:
                    return search_query

        # If no indicator found, try to extract a question
        question_match = re.search(r'\b(what|who|where|when|why|how).*\?', clean_query)
        if question_match:
            return question_match.group(0).strip('?').strip()

        # If all else fails, remove common words and return the rest
        stop_words = ["a", "an", "the", "please", "can", "you", "could", "would", "should", "i", "we", "they", "he", "she", "it"]
        words = clean_query.split()
        filtered_words = [word for word in words if word not in stop_words]

        if filtered_words:
            return ' '.join(filtered_words)

        # If nothing works, return the original query
        return query.strip()

def main():
    chatbot = SmartChatBot()
    print("\n" + "="*50)
    print("VOICE-ENABLED SMART CHATBOT")
    print("="*50)
    print("Bot: Hello! I'm your voice-enabled assistant. How can I help you?")
    print("Bot: I can now provide weather information, perform web searches, and access an expanded knowledge base!")
    print("Bot: Try asking about the weather, searching for information, or asking me questions.")
    print("Bot: If voice recognition doesn't work, you can type your messages.")
    print("Bot: Commands you can use:")
    print("     - 'text mode' - Switch to text-only input")
    print("     - 'voice mode' - Switch back to voice input")
    print("     - 'help' - Show available commands and capabilities")
    print("     - 'exit' or 'goodbye' - End the conversation")
    print("="*50 + "\n")

    # Flag to determine if we should use voice input or text input
    text_mode = False

    # Display initial help information
    initial_help = chatbot.get_help_info()
    print(f"Bot: {initial_help}")

    while True:
        try:
            # Get user input (voice or text)
            if text_mode:
                user_input = chatbot.text_input_fallback()

                # Check if user wants to switch back to voice mode
                if user_input and user_input.lower() in ["voice mode", "voice input", "switch to voice"]:
                    print("\nBot: Switching to voice input mode. I'll listen for your commands.")
                    text_mode = False
                    continue
            else:
                user_input = chatbot.listen()

                # Check if user wants to switch to text mode
                if user_input and any(phrase in user_input.lower() for phrase in ["text mode", "text input", "switch to text"]):
                    print("\nBot: Switching to text-only mode. You can type your messages now.")
                    text_mode = True
                    continue

            if user_input:
                # Generate and speak response
                response = chatbot.get_response(user_input)
                chatbot.speak(response)

                # Check for exit command
                if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
                    print("\nBot: Goodbye! Have a great day!")
                    break
        except Exception as e:
            print(f"\nError in main loop: {e}")
            print("Please try again or type 'text mode' to switch to text input.")

    print("\n" + "="*50)
    print("CHATBOT SESSION ENDED")
    print("="*50)

if __name__ == "__main__":
    main()