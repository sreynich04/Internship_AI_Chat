import os
import requests
from bs4 import BeautifulSoup

urls = [
    "https://camtech.edu.kh/",
    "https://camtech.edu.kh/why-camtech/",
    "https://camtech.edu.kh/camtech-ai-university-purpose-innovation-asia/",
    "https://camtech.edu.kh/contacts/",
    "https://camtech.edu.kh/alumni-association/"
]

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
unique_urls = list(dict.fromkeys(urls))

# Create a folder for the split files
os.makedirs("knowledge_base", exist_ok=True)

for url in unique_urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.select_one("h1") or soup.find("title")
        raw_title = title_tag.text.strip() if title_tag else "Page"
        
        # Clean the title to make a safe file name
        safe_title = "".join(c for c in raw_title if c.isalnum() or c in (' ', '-')).strip()

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")
        cleaned_lines = []
        seen_lines = set()

        for line in raw_text.splitlines():
            cleaned_line = line.strip()
            if cleaned_line and len(cleaned_line) > 2 and cleaned_line not in seen_lines:
                seen_lines.add(cleaned_line)
                cleaned_lines.append(cleaned_line)

        # Save to individual files
        file_path = os.path.join("knowledge_base", f"{safe_title}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"URL: {url}\n\n" + "\n".join(cleaned_lines))

        print(f"Saved: {safe_title}.txt")

    except Exception as e:
        print(f"Error scraping {url}: {e}")