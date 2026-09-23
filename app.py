import os
import re
import requests
from groq import Groq
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from chat_storage import init_db, save_to_history_file, get_history, log_recommendation
from recommendation_engine import rank_majors

load_dotenv(override=True)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT") or 7860)
MCP_PROMPT_FILE = "mcp_prompt.txt"
KNOWLEDGE_DIR = "knowledge_base"

app = Flask(__name__, static_folder='.', static_url_path='')



# Safely initialize database on application start
try:
    init_db()
except Exception as e:
    print(f"Warning: Database initialization error: {e}")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def sanitize_khmer_text(text: str) -> str:
    """Replaces hallucinated Thai tokens with standard Khmer terminology."""
    if not text:
        return ""
    thai_to_khmer_map = {
        "หลักสูตร": "កម្មវិធីសិក្សា",
        "มหาวิทยาลัย": "សាកលវិទ្យាល័យ",
        "วิชา": "មុខวิជ្ជា",
        "สมัคร": "ចុះឈ្មោះ",
    }
    for thai_word, khmer_word in thai_to_khmer_map.items():
        text = text.replace(thai_word, khmer_word)
    return text.strip()


def load_knowledge_base(user_message):
    """
    Content & Intent-Aware Retriever:
    Scans filenames and file contents to retrieve comprehensive context for any 
    academic, administrative, facility, tuition, or general inquiry.
    """
    if not os.path.exists(KNOWLEDGE_DIR):
        return "Knowledge base unavailable."

    user_msg_lower = user_message.lower()
    query_tokens = set(re.findall(r'\b\w{3,}\b', user_msg_lower))
    
    stop_words = {"camtech", "university", "about", "what", "how", "can", "does", "have", "with", "from", "that", "this", "tell", "know", "want"}
    filtered_tokens = query_tokens - stop_words

    all_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".txt")]
    file_scores = {}

    for file_name in all_files:
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_lower = content.lower()

                score = 0
                for token in filtered_tokens:
                    if token in file_name.lower():
                        score += 5
                    score += content_lower.count(token)

                # Intent 1: Program & Fee Queries
                if any(k in user_msg_lower for k in ["major", "majors", "program", "programs", "undergraduate", "bachelor", "bachelors", "degree", "tuition", "fee", "cost", "price"]):
                    if "bachelor" in file_name.lower():
                        score += 10

                # Intent 2: Campus, Location & Address Queries
                if any(k in user_msg_lower for k in ["location", "located", "where", "address", "map", "contact"]):
                    if any(x in content_lower for x in ["chroy chongvar", "phnom penh", "street", "location", "address"]):
                        score += 15

                # Intent 3: Facilities, Labs & Scholarships Queries
                if any(k in user_msg_lower for k in ["facility", "facilities", "lab", "labs", "maker", "scholarship", "campus"]):
                    if any(x in file_name.lower() for x in ["why", "campus", "prospectus", "bachelor"]) or "lab" in content_lower:
                        score += 10

                # Baseline weight for primary prospectus file
                if "bachelor" in file_name.lower():
                    score += 2

                file_scores[file_name] = (score, content)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    sorted_files = sorted(file_scores.items(), key=lambda x: x[1][0], reverse=True)

    selected_contents = []
    total_chars = 0
    MAX_CHAR_BUDGET = 8500

    for file_name, (score, content) in sorted_files:
        if total_chars >= MAX_CHAR_BUDGET:
            break
        
        char_limit = 7500 if "bachelor" in file_name.lower() else 3000
        truncated_content = content[:char_limit].strip()
        
        selected_contents.append(f"--- DOCUMENT: {file_name} ---\n{truncated_content}")
        total_chars += len(truncated_content)

    return "\n\n".join(selected_contents)


def load_mcp_prompt():
    """Reads the system prompt from an external text file."""
    if os.path.exists(MCP_PROMPT_FILE):
        with open(MCP_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return "You are UniGuide, the official AI Advisory Assistant for CamTech University in Phnom Penh, Cambodia."


# --- CORE LOGIC ---
def generate_response(user_message, history, session_id="session_default"):
    if not client:
        return "Error: Groq API Key not found."

    # 1. Instant Interceptor for Simple Greetings
    clean_msg = user_message.strip().lower().strip("!.,?")
    greetings_list = ["hi", "hello", "hey", "good morning", "good afternoon", "greetings", "suostei"]
    if clean_msg in greetings_list:
        return (
            "👋 Hello! Welcome to CamTech University's AI Advisory Assistant.\n\n"
            "I can help you with:\n"
            "• Undergraduate Majors & Degree Programs\n"
            "• Tuition Fees, Scholarships & Admissions\n"
            "• Campus Facilities, Labs & Student Life\n"
            "• Career Pathways & Major Recommendations\n\n"
            "How can I assist you today? Feel free to ask a question or tell me about your career goals!"
        )

    system_prompt = load_mcp_prompt()
    general_knowledge = load_knowledge_base(user_message)

    # 2. Vector Ranking & Persona Aggregation (Safeguarded)
    vector_results = []
    try:
        past_user_messages = [msg.get("content", "") for msg in history if msg.get("role") == "user"]
        past_user_messages.append(user_message)
        aggregated_user_persona = " ".join(past_user_messages)
        vector_results = rank_majors(aggregated_user_persona, top_k=2)
    except Exception as e:
        print(f"Vector ranking warning: {e}")
        aggregated_user_persona = user_message

    top_major = vector_results[0]['major'] if vector_results else "None"
    top_score = vector_results[0]['similarity_score'] if vector_results else 0.0
    mode = "DISCOVERY" if top_score < 35.0 else "RECOMMENDATION"

    try:
        log_recommendation(session_id, aggregated_user_persona, top_major, top_score, mode)
    except Exception as e:
        print(f"Logging recommendation warning: {e}")

    # 3. Dynamic Mode Instructions
    if mode == "DISCOVERY":
        ml_decision_context = f"""
        CONFIDENCE STATUS: LOW ({top_score:.2f}%)
        Top Preliminary Signals: {[r['major'] for r in vector_results]}
        
        INSTRUCTIONS FOR ASSISTANT:
        1. DIRECT GENERAL & ADMINISTRATIVE QUESTIONS (Tuition, Facilities, Admissions, Contacts, Campus Info):
           - Answer directly, completely, and accurately using the provided GENERAL FAQ CONTEXT.
           - Do NOT refuse to answer if relevant details exist anywhere in the context.

        2. EXHAUSTIVE LISTING RULE FOR MAJORS/PROGRAMS:
           - When asked about available undergraduate majors or programs, you MUST list ALL 10 majors present in the context without omitting any:
             1. AI and Data Science
             2. Architecture
             3. Risk Management and Business Intelligence
             4. Cyber Security
             5. Robotics and AI / Automation Engineering
             6. Software Engineering
             7. Interior Design
             8. Educational Technology
             9. Innovation & Entrepreneurship
             10. Media and Communication Technology

        3. PROFILE DISCOVERY (When user shares personal interests or vague career goals):
           - Do NOT make a definitive major recommendation yet.
           - Acknowledge their interest naturally and ask 1-2 brief follow-up questions to gather more details.
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
        Whenever you recommend a major, explicitly display the match confidence score:
        **[Major Name]** – *Match Confidence: [Score]%*
        """

    # 4. Master Prompt Matrix
    full_system_prompt = f"""{system_prompt}

{ml_decision_context}

CRITICAL RESPONSE GUIDELINES:
1. EXHAUSTIVE MAJORS RULE: Whenever the user asks to see available majors, undergraduate programs, or faculties, you MUST explicitly list ALL 10 degree programs below without omitting or summarizing any:
   1. AI and Data Science
   2. Architecture
   3. Risk Management and Business Intelligence
   4. Cyber Security
   5. Robotics and Automation Engineering
   6. Software Engineering
   7. Interior Design
   8. Educational Technology
   9. Innovation & Entrepreneurship
   10. Media and Communication Technology

2. INSTITUTIONAL BASELINE FACTS (NEVER REFUSE THESE):
   - Location: CamTech University campus is located in Chroy Chongvar Satellite City, Phnom Penh, Cambodia.
   - Scholarships: CamTech offers merit-based and need-based scholarships (up to 100%) based on National High School Exam results, academic standing, and CamTech entrance exams.
   - Tuition Rates: Undergraduate tuition ranges between $3,500 and $4,000 per year ($14,000 to $16,000 total for 4 years). Use this standard rate whenever asked about fees for any major.

3. NON-EXISTENT PROGRAMS: CamTech DOES NOT offer Civil, Mechanical, or general Electrical Engineering, Medicine, Nursing, or Law. Explicitly state they are not offered and direct users to existing related majors (e.g., Architecture, Robotics & Automation Engineering, Risk Management).

4. MULTI-PART QUESTIONS: Address EVERY component of a multi-topic query (e.g., tuition + location + scholarships + facilities) in a single structured response.

5. NEVER REFUSE VALID INSTITUTIONAL DATA: Do NOT say "I don't have that exact information" when asked about location, scholarships, tuition, contacts, or facilities. Rely on the baseline facts above and the GENERAL FAQ CONTEXT below.
--- GENERAL FAQ CONTEXT ---
{general_knowledge}
"""

    messages = [{"role": "system", "content": full_system_prompt}]
    for msg in history:
        raw_role = str(msg.get("role", "")).lower().strip()
        role = "assistant" if raw_role in ["bot", "model", "assistant"] else "user"
        content = msg.get("content") or msg.get("message") or ""
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    # 5. Groq Model Cascade Fallback Execution
    models_to_try = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b"
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            raw_output = completion.choices[0].message.content
            if raw_output:
                return sanitize_khmer_text(raw_output)
        except Exception as e:
            print(f"⚠️ Groq API model {model_name} failed: {e}")
            last_error = e

    return f"Advisory Error: Unable to reach Groq LLM models. Details: {str(last_error)}"


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
        try:
            chat_history = get_history(session_id)
        except Exception as e:
            print(f"Get history error: {e}")
            chat_history = []

    if not user_message:
        return jsonify({'error': 'Request missing `message` field.'}), 400
    if not client:
        return jsonify({'error': 'GROQ_API_KEY is not configured.'}), 500

    answer = generate_response(user_message, chat_history, session_id)
    
    try:
        save_to_history_file(session_id, "user", user_message)
        save_to_history_file(session_id, "assistant", answer)
    except Exception as e:
        print(f"Save history error: {e}")

    return jsonify({'response': answer})


@app.route('/telegram', methods=['POST'])
def telegram_webhook():
    payload = request.get_json(silent=True) or {}
    
    if "message" in payload:
        message = payload["message"]
        chat_id = str(message["chat"]["id"])
        user_text = message.get("text", "")

        if user_text:
            try:
                history = get_history(chat_id)
            except Exception as e:
                print(f"Telegram history fetch error: {e}")
                history = []

            bot_reply = generate_response(user_text, history, session_id=chat_id)

            try:
                save_to_history_file(chat_id, "user", user_text)
                save_to_history_file(chat_id, "assistant", bot_reply)
            except Exception as e:
                print(f"Telegram history save error: {e}")

            if TELEGRAM_BOT_TOKEN:
                telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                requests.post(telegram_url, json={"chat_id": chat_id, "text": bot_reply})

    return "OK", 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)