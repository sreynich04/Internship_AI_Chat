import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- LAZY MODEL LOADING ---
_embedding_model = None

def get_embedding_model():
    """Initializes and returns the SentenceTransformer model on demand."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

KNOWLEDGE_DIR = "knowledge_base"
CACHE_FILE = "embeddings_cache.npy"

KNOWN_MAJORS = [
    "AI and Data Science",
    "Architecture",
    "Business Intelligence",
    "Cyber Security",
    "Robotics and AI",
    "Software Engineering",
    "Educational Technology",
    "Interior Design",
    "Innovation & Entrepreneurship",
    "Media and Communication Technology"
]

# Updated MAJOR_KEYWORDS using distinct single words and variations
MAJOR_KEYWORDS = {
    "Software Engineering": [
        "code", "coding", "program", "programming", "software", "developer", "fullstack", 
        "backend", "frontend", "api", "apis", "git", "database", "databases", "app", "apps", "web"
    ],
    "AI and Data Science": [
        "ai", "data", "machine learning", "statistics", "analytics", "python", "model", 
        "models", "neural", "big data", "deep learning", "math", "mathematics", "quantitative"
    ],
    "Architecture": [
        "building", "buildings", "blueprint", "blueprints", "drafting", "structural", 
        "structure", "structures", "civil", "cad", "construction", "urban", "skyscraper", "skyscrapers"
    ],
    "Cyber Security": [
        "security", "hacking", "hacker", "hackers", "network", "networks", "cyber", 
        "firewall", "firewalls", "encryption", "malware", "defense", "intrusion"
    ],
    "Robotics and AI": [
        "robot", "robots", "robotic", "robotics", "hardware", "sensor", "sensors", 
        "automation", "mechatronics", "microcontroller", "microcontrollers", "bot", "bots", "actuator"
    ],
    "Business Intelligence": [
        "business", "finance", "financial", "market", "markets", "corporate", 
        "bi", "strategy", "strategies", "economics", "reports", "enterprise"
    ],
    "Interior Design": [
        "interior", "decor", "decorating", "room", "rooms", "furniture", "furnishing", 
        "aesthetic", "aesthetics", "indoor", "styling", "decorations"
    ],
    "Educational Technology": [
        "education", "educational", "teaching", "classroom", "classrooms", "pedagogy", 
        "e-learning", "edtech", "school", "schools", "instructional"
    ],
    "Innovation & Entrepreneurship": [
        "startup", "startups", "venture", "ventures", "pitch", "pitching", "investor", 
        "investors", "entrepreneur", "entrepreneurship", "prototype", "funding"
    ],
    "Media and Communication Technology": [
        "media", "video", "graphics", "journalism", "broadcasting", "content", 
        "multimedia", "streaming", "audio"
    ]
}

def clean_scraped_text(text: str) -> str:
    """Strips URLs, navigation headers, and web scraping noise."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'(Home|Bachelor\'s Programs|CamTech University|For more details)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_majors_from_text(file_content: str) -> dict:
    extracted = {}
    pattern = r'(' + '|'.join([re.escape(m) for m in KNOWN_MAJORS]) + r')'
    chunks = re.split(pattern, file_content)
    
    for i in range(1, len(chunks), 2):
        major_name = chunks[i].strip()
        raw_body = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
        cleaned_body = clean_scraped_text(raw_body)
        if cleaned_body:
            extracted[major_name] = f"{major_name}: {cleaned_body[:1000]}"
            
    return extracted

def get_cached_embeddings(major_texts: list):
    """Loads embeddings from disk if available and valid; otherwise encodes and saves."""
    if os.path.exists(CACHE_FILE):
        try:
            cached = np.load(CACHE_FILE)
            if len(cached) == len(major_texts):
                return cached
        except Exception:
            pass
    
    model = get_embedding_model()
    embeddings = model.encode(major_texts)
    np.save(CACHE_FILE, embeddings)
    return embeddings

def load_and_embed_majors():
    major_names = []
    major_texts = []
    major_docs = {}

    target_file = os.path.join(KNOWLEDGE_DIR, "Bachelors Programs.txt")
    if os.path.exists(target_file):
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            extracted = extract_majors_from_text(content)
            for name, text in extracted.items():
                major_names.append(name)
                major_texts.append(text)
                major_docs[name] = text

    if not major_texts:
        return [], [], {}

    major_embeddings = get_cached_embeddings(major_texts)
    return major_names, major_embeddings, major_docs

def calculate_keyword_boost(user_text: str, major_name: str) -> float:
    user_text_lower = user_text.lower()
    major_lower = major_name.lower()
    boost = 0.0

    # OPTIMIZATION 1: Direct title match boost (+0.35)
    if major_lower in user_text_lower:
        boost += 0.35

    # OPTIMIZATION 2: Keyword weight (0.10 per match, capped at 0.40)
    keywords = MAJOR_KEYWORDS.get(major_name, [])
    if keywords:
        matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', user_text_lower))
        boost += min(matches * 0.10, 0.40)

    return boost

def rank_majors(user_profile_text: str, top_k: int = 3) -> list:
    major_names, major_embeddings, major_docs = load_and_embed_majors()

    if len(major_names) == 0:
        return []

    model = get_embedding_model()
    user_vector = model.encode([user_profile_text])
    cosine_scores = cosine_similarity(user_vector, major_embeddings)[0]

    final_scores = []
    for idx, major in enumerate(major_names):
        base_score = float(cosine_scores[idx])
        boost = calculate_keyword_boost(user_profile_text, major)
        combined_score = min(base_score + boost, 1.0)
        final_scores.append(combined_score)

    ranked_indices = np.argsort(final_scores)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        results.append({
            "major": major_names[idx],
            "similarity_score": round(final_scores[idx] * 100, 2),
            "content": major_docs[major_names[idx]]
        })

    return results