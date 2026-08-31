import os
import requests
from bs4 import BeautifulSoup

urls = [
    "https://camtech.edu.kh/",
    "https://camtech.edu.kh/why-camtech/",
    "https://camtech.edu.kh/camtech-ai-university-purpose-innovation-asia/",
    "https://camtech.edu.kh/contacts/",
    "https://camtech.edu.kh/alumni-association/",
    "https://camtech.edu.kh/news-events/",
    "https://camtech.edu.kh/jobs/",
    "https://camtech.edu.kh/campus-life/",
    "https://heyzine.com/flip-book/c14349bd87.html",
    "https://camtech.edu.kh/how-to-apply/",
    "https://camtech.edu.kh/student-exchange-programs/",
    "https://camtech.edu.kh/bp/",
    "https://camtech.edu.kh/masters-and-phd-programs/",
    "https://camtech.edu.kh/engineering-2/",
    "https://camtech.edu.kh/sustainable-built-environment/",
    "https://camtech.edu.kh/arts-humanities-and-social-science/",
    "https://camtech.edu.kh/business-and-management/",
    "https://camtech.edu.kh/applied-science/",
    "https://camtech.edu.kh/school-of-continuing-education/",
    "https://camtech.edu.kh/academic-calendar-for-graduate-school/",
    "https://camtech.edu.kh/ai-forums/",
    "https://camtech.edu.kh/centers/",
    "https://camtech.edu.kh/publications/",
    "https://camtech.edu.kh/projects/",
    "https://camtech.edu.kh/wp-content/uploads/2022/12/Outline-Cyber-Essentials-Workshop12-01-2023-3-3.pdf",
    "https://camtech.edu.kh/seminars-conferences/",
    "https://camtech.edu.kh/industry-linkage/",
    "https://camtech.edu.kh/industrial-partners/",
    "https://camtech.edu.kh/university-school-collaboration/",
    "https://camtech.edu.kh/our-donors-and-sponsors/",
    "https://camtech.edu.kh/endowment-fund",
    "https://camtech.edu.kh/facts-and-figures/"
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