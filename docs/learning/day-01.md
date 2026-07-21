1. Why are vectors useful for search?
 - Every document is represented as a point on a giant map.
 - Instead of storing only the words themselves, an embedding model converts text into a list of numbers (a vector) that captures its meaning.
 - That's why vector search finds relevant information even when the wording changes.
2. Why “blood pressure medicine” may match “lisinopril”?
 - Embedding models learn relationships between words.
 - It learns statistical relationships.
3. Why query and passage embeddings must be compatible?
 - So vectors live in the same mathematical space, and coordinates can be directly comparable.
 - Use both document embedding model and user question embedding model, so both exist in the same semantic space, and distance between them has meaning.
4. Why similarity does not necessarily equal answer confidence?
 - A high similarity score only means: "These pieces of text appear semantically related."
 - It does not mean: the document answers the question, the answer is complete, the answer is correct.
 - Embeddings capture semantic similarity, not whether a passage satisfies a specific information need.


## KEY TAKEAWAYS:
Modern RAG systems often include additional steps, this layered approach helps reduce hallucinations and improves answer quality:
 - Vector retrieval: Find semantically related passages.
 - Keyword retrieval (BM25): Ensure important terms are matched.
 - Re-ranking: Use a more powerful model to sort results by answer relevance.
 - LLM validation: Determine whether the retrieved passages actually answer the question.
 - Confidence checks: If evidence is weak or conflicting, abstain instead of guessing.

 - Vectors: 	Represent the meaning of text as numbers so semantically similar text is close together.
 - "Blood pressure medicine" → "lisinopril":	Embeddings learn that related medical concepts belong near each other in vector space, even if they use different words.
 - Compatible embeddings:	Query and document vectors must come from the same embedding model (or a purposely paired query/document encoder) so similarity comparisons are meaningful.
 - Similarity ≠ confidence:	A passage can be highly related to the topic but still fail to answer the user's question. That's why RAG systems use re-ranking, validation, and confidence thresholds before responding.
