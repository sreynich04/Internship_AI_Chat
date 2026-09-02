import os
from groq import Groq  
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from chat_storage import init_db, save_to_history_file, get_history, log_recommendation
from recommendation_engine import rank_majors

load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PORT = int(os.getenv("PORT") or 7860)
MCP_PROMPT_FILE = "mcp_prompt.txt"
KNOWLEDGE_DIR = "knowledge_base"

app = Flask(__name__, static_folder='.', static_url_path='')
init_db()

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def sanitize_khmer_text(text: str) -> str:
    """Replaces hallucinated Thai tokens with standard Khmer terminology."""
    thai_to_khmer_map = {
        "หลักสูตร": "កម្មវិធីសិក្សា",
        "มหาวิทยาลัย": "សាកលវិទ្យាល័យ",
        "วิชา": "មុខវិជ្ជា",
        "สมัคร": "ចុះឈ្មោះ",
    }
    for thai_word, khmer_word in thai_to_khmer_map.items():
        text = text.replace(thai_word, khmer_word)
    return text

def load_knowledge_base(user_message):
    """Loads specific file based on keywords for general administrative FAQs."""
    if not os.path.exists(KNOWLEDGE_DIR):
        return "Knowledge base unavailable."

    user_msg_lower = user_message.lower()

    file_routing = {
        "How to Apply": ["apply", "admission", "application", "ស្នើសុំ", "ចុះឈ្មោះ"],
        "Jobs": ["job", "career", "hiring", "employment", "ការងារ"],
        "Masters and PhD Programs": ["master", "phd", "graduate program", "postgraduate"],
        "Publications": ["publication", "paper", "journal", "research output"],
        "Why CamTech": ["why", "about", "facility", "scholarship", "employability", "អាហារូបករណ៍"],
        "Industrial Partner": ["partner", "industry partner", "collaboration"],
        "Industry-Linkage": ["linkage", "industry link"],
        "News Events - CamTech University": ["news", "event", "events"],
        "SCHOOL OF CONTINUING EDUCATION": ["continuing education", "short course"],
        "SeminarsConferences": ["seminar", "conference"],
        "Student Exchange Programs": ["exchange", "study abroad"],
        "University School Collaboration": ["school collaboration", "high school"],
    }

    selected_files = []
    for filename, keywords in file_routing.items():
        if any(keyword in user_msg_lower for keyword in keywords):
            selected_files.append(filename)

    if selected_files:
        combined_text = []
        for file_name in selected_files:
            file_path = os.path.join(KNOWLEDGE_DIR, f"{file_name}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_text.append(f"--- {file_name} ---\n{f.read()}")
        if combined_text:
            return "\n\n".join(combined_text)

    default_path = os.path.join(KNOWLEDGE_DIR, "Why CamTech.txt")
    if os.path.exists(default_path):
        with open(default_path, 'r', encoding='utf-8') as f:
            return f"--- Why CamTech ---\n{f.read()}"

    return "General CamTech University information."

def load_mcp_prompt():
    """Reads the system prompt from an external text file."""
    if os.path.exists(MCP_PROMPT_FILE):
        with open(MCP_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return "You are a helpful academic recommendation assistant for CamTech University."

# --- CORE LOGIC ---
def generate_response(user_message, history, session_id="session_default"):
    if not client:
        return "Error: Groq API Key not found."

    system_prompt = load_mcp_prompt()
    general_knowledge = load_knowledge_base(user_message)

    # 1. Aggregate past history to form full user persona vector
    past_user_messages = [msg.get("content", "") for msg in history if msg.get("role") == "user"]
    past_user_messages.append(user_message)
    aggregated_user_persona = " ".join(past_user_messages)

    # 2. Compute similarity & ranking
    vector_results = rank_majors(aggregated_user_persona, top_k=2)

    top_major = vector_results[0]['major'] if vector_results else "None"
    top_score = vector_results[0]['similarity_score'] if vector_results else 0.0
    mode = "DISCOVERY" if top_score < 35.0 else "RECOMMENDATION"

    # 3. CALL LOGGER: Log analytics to SQLite database
    log_recommendation(session_id, aggregated_user_persona, top_major, top_score, mode)

    # 4. Dynamic Prompting based on Confidence Threshold
    if mode == "DISCOVERY":
        ml_decision_context = f"""
        CONFIDENCE STATUS: LOW ({top_score:.2f}%)
        Top Preliminary Signals: {[r['major'] for r in vector_results]}
        
        INSTRUCTIONS FOR ASSISTANT:
        1. Do NOT make a definitive major recommendation yet.
        2. Acknowledge their interest naturally.
        3. Ask 1-2 brief follow-up questions to gather more details.
        """
    else:
        rank_1_name = vector_results[0]['major']
        rank_1_score = vector_results[0]['similarity_score']
        
        rank_2_str = ""
        if len(vector_results) > 1:
            rank_2_str = f"- Rank 2: {vector_results[1]['major']} ({vector_results[1]['similarity_score']}% Match)"

        ml_decision_context = f"""
        CONFIDENCE STATUS: HIGH ({top_score:.2f}%)
        MATHEMATICAL RANKING RESULTS:
        - Rank 1: {rank_1_name} ({rank_1_score}% Match)
        {rank_2_str}

        STRICT FORMATTING RULE:
        Whenever you mention a recommended major, you MUST explicitly write the match percentage next to the title using this exact structure:
        **[Major Name]** – *Match Confidence: [Score]%*

        Example:
        **AI and Data Science** – *Match Confidence: {rank_1_score}%*
        """

    full_system_prompt = f"""{system_prompt}

{ml_decision_context}

--- GENERAL FAQ CONTEXT ---
{general_knowledge}
"""

    messages = [{"role": "system", "content": full_system_prompt}]

    for msg in history:
        role = msg.get("role", "user")
        if role in ["bot", "model"]:
            role = "assistant"
        content = msg.get("content") or msg.get("message") or ""
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.6,
            max_tokens=1000
        )
        raw_output = completion.choices[0].message.content
        return sanitize_khmer_text(raw_output)
    except Exception as e:
        return f"Advisory Error: {str(e)}"

# --- ROUTES ---
@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get('message') or ''
    session_id = payload.get('session_id', 'session_default')

    chat_history = payload.get('history')
    if chat_history is None:
        chat_history = get_history(session_id)

    if not user_message:
        return jsonify({'error': 'Request missing `message` field.'}), 400
    if not client:
        return jsonify({'error': 'GROQ_API_KEY is not configured.'}), 500

    answer = generate_response(user_message, chat_history, session_id)
    save_to_history_file(session_id, user_message, answer)

    return jsonify({'response': answer})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)