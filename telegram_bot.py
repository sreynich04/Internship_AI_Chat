import os
import telebot
from dotenv import load_dotenv

# Import core backend functions from app.py pipeline
from app import generate_response
from chat_storage import get_history, save_to_history_file

load_dotenv()

# Load Telegram Token from environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8960580452:AAG3PI3zvbxUBjHzqP9fJIT-oj0zRj4I248")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "👋 Welcome to the official CamTech University Advisory Assistant!\n\n"
        "I can help you with:\n"
        "• Academic Major Recommendations\n"
        "• Admissions, Tuition & Scholarships\n"
        "• Campus Facilities, Exchange Programs & Events\n\n"
        "Tell me about your career interests or ask any question about CamTech!"
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(func=lambda message: True)
def handle_telegram_chat(message):
    user_text = message.text.strip()
    
    # 1. Map unique Telegram Chat ID as the session_id for isolated chat history
    session_id = f"tg_{message.chat.id}"

    # 2. Indicate typing status while processing vector search + Groq LLM
    bot.send_chat_action(message.chat.id, 'typing')

    try:
        # 3. Retrieve conversation history for this Telegram user from SQLite/Turso
        chat_history = get_history(session_id)

        # 4. Generate answer using complete app.py pipeline
        bot_response = generate_response(user_text, chat_history, session_id)

        # 5. Save user message and AI response back to chat storage
        save_to_history_file(session_id, "user", user_text)
        save_to_history_file(session_id, "assistant", bot_response)

        # 6. Send formatted reply back to user
        try:
            bot.reply_to(message, bot_response, parse_mode="Markdown")
        except Exception:
            # Fallback to plain text if Markdown parsing fails on raw LLM symbols
            bot.reply_to(message, bot_response)

    except Exception as e:
        print(f"Error handling Telegram message: {e}")
        bot.reply_to(message, "I encountered an error processing your request. Please try again.")

if __name__ == "__main__":
    print("🤖 Deleting existing active webhooks...")
    bot.remove_webhook()
    print("🤖 CamTech Telegram Bot successfully connected to app.py logic...")
    bot.infinity_polling()