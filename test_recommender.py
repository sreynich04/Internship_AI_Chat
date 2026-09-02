from recommendation_engine import rank_majors

# Example student input aggregating interests from past chat turns
student_profile = "I enjoy drawing, 3D graphic design, UI layout, and basic web coding."

recommendations = rank_majors(student_profile, top_k=3)

print("\n--- VECTOR RECOMMENDATION RESULTS ---")
for index, rec in enumerate(recommendations, 1):
    print(f"Rank {index}: {rec['major']} | Match Score: {rec['similarity_score']}%")
# Add this inside test_recommender.py to inspect text chunks
for rec in recommendations:
    print(f"\n--- {rec['major']} TEXT CHUNK ---")
    print(rec['content'][:300])  # Shows first 300 characters
