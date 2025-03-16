# Voice-Enabled Smart Chatbot

A Python-based intelligent chatbot that supports voice interactions, natural language processing, and basic task automation.

## Features

- Voice input and output capabilities
- Natural Language Processing using NLTK
- Basic task execution:
  - Time queries
  - Simple calculations
  - Weather information (placeholder)
  - Web search (placeholder)
- Response generation using text preprocessing and similarity matching
- Pre-defined responses for common interactions (greetings, farewells, thanks)

## Requirements

```bash
nltk
SpeechRecognition
pyttsx3
scikit-learn
pyaudio
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. The first run will download necessary NLTK data automatically

## Usage

Run the chatbot:
```bash
python chatbot/chatbot.py
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
- Exit: "bye", "goodbye", "exit", "quit"

## Project Structure

```
Chatbot/
├── chatbot.py      # Main chatbot implementation
└── requirements.txt # Project dependencies
```

## Technical Details

- Uses `SpeechRecognition` for voice input
- `pyttsx3` for text-to-speech conversion
- NLTK for text preprocessing and understanding
- Implements basic NLP techniques:
  - Tokenization
  - Stopword removal
  - Lemmatization
  - TF-IDF vectorization

## Future Enhancements

- Implement weather information retrieval
- Add web search capabilities
- Expand knowledge base
- Improve response generation using advanced NLP techniques
