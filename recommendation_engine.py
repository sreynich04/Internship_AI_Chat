import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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

MAJOR_KEYWORDS = {
    "Software Engineering": ["code", "coding", "programming", "web", "app", "software", "developer", "javascript", "python", "fullstack", "ui", "ux"],
    "AI and Data Science": ["ai", "data", "machine learning", "statistics", "analytics", "python", "model", "neural", "numbers", "big data", "deep learning", "math", "maths", "mathematics", "quantitative"],
    "Architecture": ["building", "architecture", "structure", "3d", "spatial", "drafting", "blueprint", "construction"],
    "Cyber Security": ["security", "hacking", "network", "cyber", "firewall", "encryption", "defense"],
    "Robotics and AI": ["robot", "robotics", "hardware", "sensor", "automation", "mechatronics"],
    "Business Intelligence": ["business", "finance", "market", "management", "strategy", "bi", "analytics", "economics", "data", "reports"],
    "Interior Design": ["interior", "decor", "design", "furniture", "space", "aesthetics"],
    "Media and Communication Technology": ["media", "graphics", "video", "communication", "content", "journalism"]
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
    
    embeddings = embedding_model.encode(major_texts)
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

    # FIXED: Utilizes get_cached_embeddings to eliminate redundant processing
    major_embeddings = get_cached_embeddings(major_texts)
    return major_names, major_embeddings, major_docs

def calculate_keyword_boost(user_text: str, major_name: str) -> float:
    keywords = MAJOR_KEYWORDS.get(major_name, [])
    if not keywords:
        return 0.0
    
    user_text_lower = user_text.lower()
    matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', user_text_lower))
    return min(matches * 0.05, 0.15)

def rank_majors(user_profile_text: str, top_k: int = 3) -> list:
    major_names, major_embeddings, major_docs = load_and_embed_majors()

    if len(major_names) == 0:
        return []

    user_vector = embedding_model.encode([user_profile_text])
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