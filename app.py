import os
import json
from groq import Groq  
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
MCP_PROMPT_FILE = "mcp_prompt.txt"
HISTORY_FILE = "chat_history.json"
KNOWLEDGE_DIR = "knowledge_base"

app = Flask(__name__, static_folder='.', static_url_path='')

# Initialize the Groq Client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

KNOWLEDGE_DIR = "knowledge_base"

def load_knowledge_base(user_message):
    """Loads specific file based on keywords, or loads all files as a fallback."""
    if not os.path.exists(KNOWLEDGE_DIR):
        return "Knowledge base unavailable."

    user_msg_lower = user_message.lower()
    
    file_routing = {
        "Contacts": ["contact", "phone", "email", "address", "location", "reach", "ទំនាក់ទំនង", "លេខទូរស័ព្ទ", "អាសយដ្ឋាន", "អ៊ីមែល"],
        "Alumni Association": ["alumni", "graduate", "past students", "សិស្សចាស់", "អតីតនិស្សិត"],
        "Why CamTech": ["why", "about", "facility", "scholarship", "employability", "faculty", "អាហារូបករណ៍", "ហេតុអ្វី", "សម្ភារៈ"],
        "CamTech AI University  Purpose Innovation Asia": ["ai", "phd", "purpose", "ethics", "research", "បញ្ញាសិប្បនិម្មិត", "ស្រាវជ្រាវ"],
        "Home - CamTech University": ["apply", "admission", "general", "degree", "ចុះឈ្មោះ", "ស្នើសុំ"] 
    }

    selected_file = None
    for filename, keywords in file_routing.items():
        if any(keyword in user_msg_lower for keyword in keywords):
            selected_file = filename
            break

    # 1. If a keyword matched, load that specific file
    if selected_file:
        file_path = os.path.join(KNOWLEDGE_DIR, f"{selected_file}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f"--- {selected_file} ---\n{f.read()}"

    # 2. FALLBACK: If no keyword matched (e.g. unknown Khmer phrase), load ALL files so no information is missed!
    combined_text = []
    for file_name in os.listdir(KNOWLEDGE_DIR):
        if file_name.endswith(".txt"):
            full_path = os.path.join(KNOWLEDGE_DIR, file_name)
            with open(full_path, 'r', encoding='utf-8') as f:
                combined_text.append(f"--- {file_name} ---\n{f.read()}")

    return "\n\n".join(combined_text) if combined_text else "No relevant data found."

def load_mcp_prompt():
    """Reads the system prompt from an external text file"""
    if os.path.exists(MCP_PROMPT_FILE):
        with open(MCP_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return "You are a helpful university AI assistant."

def save_to_history_file(session_id, user_message, ai_response):
    """Saves messages grouped under their specific session ID"""
    history_data = {}
    
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            pass

    # Ensure structure is a dictionary
    if not isinstance(history_data, dict):
        history_data = {}

    # Initialize new session array if it doesn't exist
    if session_id not in history_data:
        history_data[session_id] = []

    # Append interaction to this session
    history_data[session_id].append({"role": "user", "content": user_message})
    history_data[session_id].append({"role": "assistant", "content": ai_response})
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)

# --- CORE LOGIC ---
def generate_response(user_message, history):
    if not client:
        return "Error: Groq API Key not found."

    # 1. Dynamically load data so it auto-updates without restarting the server!
    system_prompt = load_mcp_prompt()

    # Pass the user_message so it knows which file to search for!
    knowledge_base = load_knowledge_base(user_message)

    # DEBUG CUZ DID NOT UPDATE THE KNOWLEDGE BASE WHEN I UPDATED THE FILE
    print("\n--- DEBUG: WHAT PYTHON SEES --- \n", knowledge_base, "\n------------------------------\n")

    # 2. Combine System Prompt with Knowledge Base
    full_system_prompt = f"{system_prompt}\n\n--- KNOWLEDGE BASE ---\n{knowledge_base}"

    # 3. Initialize the messages array
    messages = [{"role": "system", "content": full_system_prompt}]

    # 4. Append frontend chat history
    for msg in history:
        messages.append({
            "role": msg.get("role"), 
            "content": msg.get("content")
        })

    # 5. Append the newest user question
    messages.append({"role": "user", "content": user_message})
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",  
            messages=messages, 
            temperature=0.7,
            max_tokens=2048
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            return "⚠️ The Groq engine is busy! Please wait a few seconds and try again."
        return f"Advisory Error: {str(e)}"

# --- ROUTES ---
@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get('message') or ''
    chat_history = payload.get('history', []) 
    session_id = payload.get('session_id', 'session_default') # Read session_id

    if not user_message:
        return jsonify({'error': 'Request missing `message` field.'}), 400
    if not client:
        return jsonify({'error': 'GROQ_API_KEY is not configured.'}), 500

    answer = generate_response(user_message, chat_history)
    
    # Save using session_id
    save_to_history_file(session_id, user_message, answer)

    return jsonify({'response': answer})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    app.run(host='0.0.0.0', port=port)