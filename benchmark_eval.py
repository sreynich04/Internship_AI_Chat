# Batch testing script to evaluate the recommendation engine's accuracy against a 
# 50-case benchmark dataset and generate a visual presentation dashboard.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from recommendation_engine import rank_majors

# Expanded 50-case benchmark test suite covering all 10 major programs
TEST_DATASET = [
    # Software Engineering
    {"input": "I like drawing, 3D graphic design, UI layout, and basic web coding.", "target": "Software Engineering"},
    {"input": "I want to build web applications, write Python code, and manage databases.", "target": "Software Engineering"},
    {"input": "building desktop software, backend APIs, and learning git version control.", "target": "Software Engineering"},
    {"input": "mobile app development with fullstack JavaScript frameworks.", "target": "Software Engineering"},
    {"input": "front end ui ux interface design and web development.", "target": "Software Engineering"},

    # AI and Data Science
    {"input": "I want to train machine learning models, analyze big data, and build neural networks.", "target": "AI and Data Science"},
    {"input": "machine learning, neural networks, probability, and statistics.", "target": "AI and Data Science"},
    {"input": "analyzing big data sets with mathematical models and analytics.", "target": "AI and Data Science"},
    {"input": "I love data analytics, python programming, and deep learning algorithms.", "target": "AI and Data Science"},
    {"input": "working with quantitative charts, data trends, and artificial intelligence.", "target": "AI and Data Science"},

    # Architecture
    {"input": "I love designing physical buildings, spatial layouts, blueprints, and structural models.", "target": "Architecture"},
    {"input": "drafting architectural blueprints for skyscraper construction and CAD modeling.", "target": "Architecture"},
    {"input": "building structural models, urban planning, and civil drafting.", "target": "Architecture"},
    {"input": "I want to design urban buildings, construction sites, and spatial frameworks.", "target": "Architecture"},
    {"input": "spatial CAD modeling for large scale building and construction projects.", "target": "Architecture"},

    # Interior Design
    {"input": "I focus on room decor, interior spatial design, furniture aesthetics, and layouts.", "target": "Interior Design"},
    {"input": "decorating rooms, furniture layout, and spatial indoor aesthetics.", "target": "Interior Design"},
    {"input": "styling house interiors, home decoration, and room furnishing.", "target": "Interior Design"},
    {"input": "selecting wall colors, furniture aesthetics, and room layout decor.", "target": "Interior Design"},
    {"input": "commercial indoor aesthetics, room design, and interior space styling.", "target": "Interior Design"},

    # Cyber Security
    {"input": "I am interested in network security, stopping ethical hackers, and data encryption.", "target": "Cyber Security"},
    {"input": "ethical hacking, network defense, firewalls, and cyber threats.", "target": "Cyber Security"},
    {"input": "preventing cyber attacks and applying strong data encryption security.", "target": "Cyber Security"},
    {"input": "protecting servers against malicious malware, intruders, and hackers.", "target": "Cyber Security"},
    {"input": "securing cloud databases against network intrusion and data breaches.", "target": "Cyber Security"},

    # Business Intelligence
    {"input": "I want to analyze business data, financial trends, and build market strategies.", "target": "Business Intelligence"},
    {"input": "corporate business management, market strategy, and finance analysis.", "target": "Business Intelligence"},
    {"input": "BI reporting, business growth strategies, and market analytics.", "target": "Business Intelligence"},
    {"input": "analyzing financial reports to guide corporate business decisions.", "target": "Business Intelligence"},
    {"input": "corporate economics, enterprise data reports, and business intelligence.", "target": "Business Intelligence"},

    # Robotics and AI
    {"input": "I like working with physical hardware sensors, microcontrollers, and automated bots.", "target": "Robotics and AI"},
    {"input": "building hardware robots with sensors, actuators, and electronics.", "target": "Robotics and AI"},
    {"input": "mechatronics, automated microcontrollers, and robotic hardware.", "target": "Robotics and AI"},
    {"input": "programming robotic arms, automated machinery, and sensors.", "target": "Robotics and AI"},
    {"input": "building autonomous vehicles with integrated mechatronic hardware.", "target": "Robotics and AI"},

    # Educational Technology
    {"input": "teaching tools, school e-learning platforms, and educational tech.", "target": "Educational Technology"},
    {"input": "digital classrooms, online learning tools, and modern teaching methods.", "target": "Educational Technology"},
    {"input": "instructional pedagogy and educational software platforms for schools.", "target": "Educational Technology"},
    {"input": "how to integrate technology tools into classroom teaching.", "target": "Educational Technology"},
    {"input": "e-learning curriculum development and modern educational technology.", "target": "Educational Technology"},

    # Innovation & Entrepreneurship
    {"input": "launching tech startups and pitching ideas to venture capital investors.", "target": "Innovation & Entrepreneurship"},
    {"input": "entrepreneurship, building new startup products, and business models.", "target": "Innovation & Entrepreneurship"},
    {"query": "how to start a tech company, prototype ideas, and pitch investors.", "target": "Innovation & Entrepreneurship"},
    {"input": "venture funding, prototyping innovative products, and starting a venture.", "target": "Innovation & Entrepreneurship"},
    {"input": "business model discovery and pitch strategy for new entrepreneurial startups.", "target": "Innovation & Entrepreneurship"},

    # Media and Communication Technology
    {"input": "video editing, graphic design, media tech, and content creation.", "target": "Media and Communication Technology"},
    {"input": "digital journalism, media production, content creation, and broadcasting.", "target": "Media and Communication Technology"},
    {"input": "graphics design, multimedia streaming, social content, and video production.", "target": "Media and Communication Technology"},
    {"input": "producing digital video media content and broadcasting technology.", "target": "Media and Communication Technology"},
    {"input": "communication technology, journalism, media production, and audio editing.", "target": "Media and Communication Technology"}
]

def generate_dashboard(df, y_true, y_pred, top_1_accuracy, mrr):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Chart 1: Per-Major Top-1 Accuracy ---
    major_acc = df.groupby("Target")["Top_1_Pass"].mean().reset_index()
    major_acc["Top_1_Pass"] = major_acc["Top_1_Pass"] * 100

    palette = ["#2ecc71" if acc == 100 else "#e74c3c" for acc in major_acc["Top_1_Pass"]]

    bars = sns.barplot(data=major_acc, x="Top_1_Pass", y="Target", ax=axes[0], palette=palette)
    axes[0].set_title("Top-1 Recommendation Accuracy by Major (%)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Accuracy (%)", fontsize=12)
    axes[0].set_ylabel("Target Major", fontsize=12)
    axes[0].set_xlim(0, 110)

    for bar in bars.patches:
        width = bar.get_width()
        axes[0].text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.0f}%", 
                     va="center", fontsize=10, fontweight="bold")

    # --- Chart 2: Confusion Matrix Heatmap ---
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title("Confusion Matrix (Ground Truth vs Predicted)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Major", fontsize=12)
    axes[1].set_ylabel("Target Major", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # --- Metric Summary Box ---
    kpi_text = f"Total Test Samples: {len(df)}  |  Top-1 Accuracy: {top_1_accuracy:.2f}%  |  Mean Reciprocal Rank (MRR): {mrr:.4f}"
    fig.text(0.5, 0.01, kpi_text, ha="center", fontsize=12, fontweight="bold", 
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=1.0))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("benchmark_dashboard.png", dpi=300)
    print("Saved presentation chart dashboard to 'benchmark_dashboard.png'!\n")

def evaluate_benchmark():
    total_queries = len(TEST_DATASET)
    reciprocal_ranks = []
    correct_top_1 = 0

    y_true = []
    y_pred = []
    results_data = []

    print("\n=== RUNNING ACCURACY BENCHMARK EVALUATION (50 TEST CASES) ===\n")

    for idx, item in enumerate(TEST_DATASET, 1):
        user_input = item.get("input") or item.get("query")
        target = item["target"]
        
        results = rank_majors(user_input, top_k=5)
        ranked_majors = [r["major"] for r in results]
        selected_major = ranked_majors[0] if ranked_majors else "None"

        y_true.append(target)
        y_pred.append(selected_major)
        
        if target in ranked_majors:
            rank = ranked_majors.index(target) + 1
            reciprocal_rank = 1.0 / rank
        else:
            rank = "Not in Top 5"
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)

        is_top_1 = (rank == 1)
        if is_top_1:
            correct_top_1 += 1
            status = "PASS (Rank 1)"
        else:
            status = f"FAIL (Ranked #{rank})"

        print(f"Test {idx:02d}: Target '{target}' -> Selected '{selected_major}' | Status: {status}")

        results_data.append({
            "Query": user_input,
            "Target": target,
            "Predicted": selected_major,
            "Top_1_Pass": is_top_1
        })

    mrr = sum(reciprocal_ranks) / total_queries
    top_1_accuracy = (correct_top_1 / total_queries) * 100

    print("\n=== BENCHMARK METRICS ===")
    print(f"Total Test Queries: {total_queries}")
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("=========================\n")

    # Generate Chart Image
    df = pd.DataFrame(results_data)
    generate_dashboard(df, y_true, y_pred, top_1_accuracy, mrr)

if __name__ == "__main__":
    evaluate_benchmark()