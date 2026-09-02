from recommendation_engine import rank_majors

# Standard benchmark test suite with expected ground-truth targets
TEST_DATASET = [
    {"input": "I like drawing, 3D graphic design, UI layout, and basic web coding.", "target": "Software Engineering"},
    {"input": "I want to build web applications, write Python code, and manage databases.", "target": "Software Engineering"},
    {"input": "I love designing physical buildings, spatial layouts, blueprints, and structural models.", "target": "Architecture"},
    {"input": "I am interested in network security, stopping ethical hackers, and data encryption.", "target": "Cyber Security"},
    {"input": "I want to analyze business data, financial trends, and build market strategies.", "target": "Business Intelligence"},
    {"input": "I like working with physical hardware sensors, microcontrollers, and automated bots.", "target": "Robotics and AI"},
    {"input": "I want to train machine learning models, analyze big data, and build neural networks.", "target": "AI and Data Science"},
    {"input": "I focus on room decor, interior spatial design, furniture aesthetics, and layouts.", "target": "Interior Design"},
]

def evaluate_benchmark():
    total_queries = len(TEST_DATASET)
    reciprocal_ranks = []
    correct_top_1 = 0

    print("\n=== RUNNING ACCURACY BENCHMARK EVALUATION ===\n")

    for idx, item in enumerate(TEST_DATASET, 1):
        user_input = item["input"]
        target = item["target"]
        
        results = rank_majors(user_input, top_k=5)
        ranked_majors = [r["major"] for r in results]
        
        if target in ranked_majors:
            rank = ranked_majors.index(target) + 1
            reciprocal_rank = 1.0 / rank
        else:
            rank = "Not in Top 5"
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)

        if rank == 1:
            correct_top_1 += 1
            status = "PASS (Rank 1)"
        else:
            status = f"FAIL (Ranked #{rank})"

        print(f"Test {idx}: Target '{target}' -> Selected '{ranked_majors[0]}' | Status: {status}")

    mrr = sum(reciprocal_ranks) / total_queries
    top_1_accuracy = (correct_top_1 / total_queries) * 100

    print("\n=== BENCHMARK METRICS ===")
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("=========================\n")

if __name__ == "__main__":
    evaluate_benchmark()