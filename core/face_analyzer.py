import torch
from insightface.app import FaceAnalysis
from config.settings import MODELS_DIR
from config.constants import MODEL_PROVIDERS, DETECTION_THRESHOLD, USE_GPU


class FaceAnalyzer:
    def __init__(self):
        self.app = None
        self.load_model()

    def load_model(self):
        """Load the face analysis model with environment-based configuration"""
        device = torch.device(
            "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
        )

        # Use available providers based on environment configuration
        available_providers = []
        for provider in MODEL_PROVIDERS:
            if provider == "CUDAExecutionProvider" and device.type == "cuda":
                available_providers.append(provider)
            elif provider == "CPUExecutionProvider":
                available_providers.append(provider)

        # Fallback to CPU if no providers are available
        if not available_providers:
            available_providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name=str(MODELS_DIR), providers=available_providers)

        self.app.prepare(
            ctx_id=0 if device.type == "cuda" else -1,
            det_thresh=DETECTION_THRESHOLD,
            det_size=(640, 640),
        )
        return self.app

    def get_faces(self, frame):
        """Extract faces from frame"""
        return self.app.get(frame) if self.app else None
