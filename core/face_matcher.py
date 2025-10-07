import cupy as cp
import numpy as np
from config.constants import SIMILARITY_THRESHOLD


class FaceMatcher:
    @staticmethod
    def cosine_similarity_cupy(A, B):
        """Calculate cosine similarity using CuPy"""
        A_norm = cp.linalg.norm(A, axis=1, keepdims=True)
        B_norm = cp.linalg.norm(B, axis=1, keepdims=True)
        similarity = cp.dot(A, B.T) / (A_norm * B_norm.T)
        return similarity

    def match(
        self, embedding, user_ids, feature_matrix, threshold=SIMILARITY_THRESHOLD
    ):
        """Match face embedding against feature matrix"""
        embedding = cp.asarray(embedding).reshape(1, -1)
        similarities = self.cosine_similarity_cupy(embedding, feature_matrix)[0]
        max_index = int(cp.argmax(similarities).item())
        max_score = similarities[max_index].item()

        sim_user_id = user_ids[max_index] if max_score >= threshold else "Unknown"
        return sim_user_id, max_score
