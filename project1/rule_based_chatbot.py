"""
Project 1: Rule-Based AI Chatbot
Decode Labs - AI Internship 2026

Architecture: Dictionary-based intent matching (O(1) lookup)
instead of if-elif ladder (O(n)). Uses keyword matching for
slightly more natural conversation flow.
"""

import random

# ---------------------------------------------------------
# KNOWLEDGE BASE
# Each key is an intent, value is a list of possible replies
# (a list lets the bot vary its responses instead of being robotic)
# ---------------------------------------------------------
responses = {
    "greeting": [
        "Hey there! How can I help you today?",
        "Hello! What's on your mind?",
    ],
    "how_are_you": [
        "I'm just code, but I'm running smoothly! How about you?",
        "Doing great, thanks for asking!",
    ],
    "name": [
        "I'm a rule-based assistant, built as part of the Decode Labs AI track.",
    ],
    "thanks": [
        "You're welcome!",
        "Anytime!",
    ],
    "help": [
        "I can chat about basics like greetings, how you're doing, or my name. Try me!",
    ],
    "weather": [
        "I can't check live weather yet — that needs an API, not just rules!",
    ],
}

# Maps keywords -> intent. This is the "matching" layer.
keyword_map = {
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "how are you": "how_are_you",
    "your name": "name",
    "who are you": "name",
    "thank you": "thanks",
    "thanks": "thanks",
    "help": "help",
    "weather": "weather",
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}
DEFAULT_REPLY = "I don't understand that yet — try saying 'help' to see what I can do."


def get_intent(clean_input: str):
    """Check if any known keyword appears in the user's input."""
    for keyword, intent in keyword_map.items():
        if keyword in clean_input:
            return intent
    return None


def get_reply(clean_input: str) -> str:
    intent = get_intent(clean_input)
    if intent:
        return random.choice(responses[intent])
    return DEFAULT_REPLY


def main():
    print("Bot: Hi! I'm a rule-based chatbot. Type 'bye' to exit.")

    while True:  # PHASE: continuous loop (the "heartbeat")
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()  # PHASE: sanitization

        if clean_input in EXIT_COMMANDS:
            print("Bot: Goodbye! 👋")
            break  # PHASE: kill command

        reply = get_reply(clean_input)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    main()