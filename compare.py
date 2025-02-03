import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example embeddings
embedding1 = np.array([0.1234, -0.5678, 0.9101, 0.2345, -0.6789, 0.8123, 0.5678, -0.2345, 0.9876, -0.6543])
embedding2 = np.array([0.2345, -0.6789, 0.8123, 0.5678, -0.2345, 0.9876, -0.6543])

# Check the length of both embeddings
len1 = len(embedding1)
len2 = len(embedding2)

# Pad or trim the smaller embedding
if len1 > len2:
    # Pad embedding2 with zeros to match embedding1's length
    embedding2 = np.pad(embedding2, (0, len1 - len2), 'constant')
elif len1 < len2:
    # Pad embedding1 with zeros to match embedding2's length
    embedding1 = np.pad(embedding1, (0, len2 - len1), 'constant')

# Now both embeddings have the same length, you can calculate cosine similarity
cos_sim = cosine_similarity([embedding1], [embedding2])

print("Cosine Similarity: ", cos_sim[0][0])
