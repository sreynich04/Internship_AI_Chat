import os
import re
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = "knowledge_base"
CACHE_FILE = "embeddings_cache.npy"

# --- HUGGING FACE INFERENCE API ---
HF_TOKEN = os.getenv("HF_TOKEN", "")

def query_hf_embeddings(texts: list) -> np.ndarray:
    """Fetches vector embeddings remotely with full safety checks for non-list JSON responses."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Priority order of Hugging Face inference endpoints
    urls = [
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    ]

    for url in urls:
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=8
            )
            
            if response.status_code == 200:
                res_json = response.json()
                
                # Check if HF returned an error dictionary instead of numerical vectors
                if isinstance(res_json, dict):
                    print(f"HF API returned dictionary/error: {res_json}")
                    continue
                
                data = np.array(res_json, dtype=np.float32)
                
                # Handle 3D token arrays: mean pool to 2D (batch_size, 384)
                if data.ndim == 3:
                    data = np.mean(data, axis=1)
                elif data.ndim == 1:
                    data = np.expand_dims(data, axis=0)

                if data.ndim == 2 and data.shape[1] == 384:
                    return data
            else:
                print(f"HF API Error {response.status_code} on {url}: {response.text}")
        except Exception as e:
            print(f"HF API Exception on {url}: {e}")

    # Fail-safe zero matrix fallback to prevent Flask 500 crashes
    return np.zeros((len(texts), 384), dtype=np.float32)

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
    if os.path.exists(CACHE_FILE):
        try:
            cached = np.load(CACHE_FILE)
            if cached.ndim == 3:
                cached = np.mean(cached, axis=1)
            if len(cached) == len(major_texts) and cached.ndim == 2:
                return cached
        except Exception:
            pass
    
    embeddings = query_hf_embeddings(major_texts)
    if np.any(embeddings):
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
        return [], np.array([]), {}

    major_embeddings = get_cached_embeddings(major_texts)
    return major_names, major_embeddings, major_docs

def calculate_keyword_boost(user_text: str, major_name: str) -> float:
    user_text_lower = user_text.lower()
    major_lower = major_name.lower()
    boost = 0.0

    if major_lower in user_text_lower:
        boost += 0.35

    keywords = MAJOR_KEYWORDS.get(major_name, [])
    if keywords:
        matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', user_text_lower))
        boost += min(matches * 0.10, 0.40)

    return boost

def rank_majors(user_profile_text: str, top_k: int = 3) -> list:
    try:
        major_names, major_embeddings, major_docs = load_and_embed_majors()

        if len(major_names) == 0:
            return []

        user_vector = query_hf_embeddings([user_profile_text])
        
        # Ensure matrix dimensions align properly
        if user_vector.ndim == 3:
            user_vector = np.mean(user_vector, axis=1)
        if major_embeddings.ndim == 3:
            major_embeddings = np.mean(major_embeddings, axis=1)

        # Calculate cosine similarity if vectors exist, otherwise default to keyword ranking
        if user_vector.ndim == 2 and major_embeddings.ndim == 2 and user_vector.shape[1] == major_embeddings.shape[1]:
            cosine_scores = cosine_similarity(user_vector, major_embeddings)[0]
        else:
            cosine_scores = np.zeros(len(major_names))

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
    except Exception as e:
        print(f"Error inside rank_majors: {e}")
        return []