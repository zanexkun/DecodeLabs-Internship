responses = {
    'hello': 'Hi there! How can I help you today?',
    'hi': 'Hello! Nice to meet you.',
    'how are you': 'I am just a program, but I am doing great, thanks for asking!',
    'what is your name': 'I am a simple rule-based chatbot built for DecodeLabs.',
    'bye': 'Goodbye! Have a great day.',
}

print("Chatbot: Hi! type exit to end the conversation.")

while True:
    user_input = input("You: ")
    user_input_cleaned = user_input.lower().strip()

    if user_input_cleaned == 'exit':
        print("Chatbot: Goodbye!")
        break

    reply = responses.get(user_input_cleaned, "I do not understand that yet.")
    print("Chatbot:", reply)
