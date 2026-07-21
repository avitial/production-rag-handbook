from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

passages = [
    "The patient has hypertension.",
    "The patient was prescribed lisinopril.",
    "Lisinopril is prescribed for hypertension.",
    "The patient has no known drug allergies.",
    "A follow-up appointment was scheduled.",
]

query = "What medicine treats the patient's blood pressure?"

passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
)

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
)

scores = passage_embeddings @ query_embedding[0]

ranked = sorted(
    zip(passages, scores),
    key=lambda item: item[1],
    reverse=True,
)

for passage, score in ranked:
    print(f"{score:.4f}: {passage}")
