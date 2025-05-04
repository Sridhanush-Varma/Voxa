# Voice-Enabled Smart Chatbot

A Python-based intelligent chatbot that supports voice interactions, natural language processing, weather information retrieval, web search capabilities, and an expanded knowledge base with advanced response generation.

## Features

- Voice input and output capabilities
- Natural Language Processing using NLTK
- Task execution:
  - Time queries
  - Simple calculations
  - Weather information retrieval using OpenWeatherMap API
  - Web search capabilities using Google Custom Search API
- Advanced response generation with sentiment analysis and context awareness
- Expanded knowledge base with facts and Q&A pairs
- Pre-defined responses for common interactions (greetings, farewells, thanks)

## Requirements

```bash
nltk
SpeechRecognition
pyttsx3
scikit-learn
pyaudio
requests
beautifulsoup4
google-api-python-client
textblob
python-dotenv
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. The first run will download necessary NLTK data automatically
4. Create a `.env` file in the root directory with your API keys:
   ```
   WEATHER_API_KEY=your_openweathermap_api_key
   GOOGLE_SEARCH_API_KEY=your_google_api_key
   SEARCH_ENGINE_ID=your_search_engine_id
   ```

## Usage

Run the chatbot:
```bash
python Voxa.py
```

The chatbot will:
1. Start listening for voice input
2. Convert speech to text
3. Process the command
4. Respond with both text and voice output

## Supported Commands

- Greetings: "hello", "hi", "hey"
- Time queries: "what's the time?"
- Calculations: "calculate 5 plus 3"
- Weather information: "what's the weather in London?", "weather forecast for New York"
- Web search: "search for latest AI developments", "find information about climate change"
- Knowledge base queries: "what is Python?", "tell me about machine learning"
- Exit: "bye", "goodbye", "exit", "quit"

## Project Structure

```
Voxa/
├── Voxa.py            # Main chatbot implementation
├── knowledge_base.json # Extended knowledge base with facts and Q&A pairs
├── requirements.txt   # Project dependencies
├── .env               # Environment variables for API keys (not tracked by git)
└── .gitignore         # Git ignore file to exclude sensitive information
```

## Technical Details

- Uses `SpeechRecognition` for voice input
- `pyttsx3` for text-to-speech conversion
- NLTK for text preprocessing and understanding
- Implements advanced NLP techniques:
  - Tokenization
  - Stopword removal
  - Lemmatization
  - TF-IDF vectorization
  - Sentiment analysis with TextBlob
  - Cosine similarity for text matching
- API Integrations:
  - OpenWeatherMap API for weather information
  - Google Custom Search API for web searches
- JSON-based knowledge base for extensible information storage
- Conversation context tracking for multi-turn interactions

## Future Enhancements

- Implement voice activity detection for better listening
- Add support for multiple languages
- Integrate with smart home devices
- Implement machine learning for continuous improvement
- Add user authentication and personalization
- Develop a graphical user interface