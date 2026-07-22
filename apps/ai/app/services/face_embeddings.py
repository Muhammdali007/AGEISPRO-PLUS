from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from app.core.config import settings

_SERVICE_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in _SERVICE_FILE.parents
        if (parent / "docker-compose.yml").exists() or (parent / "storage").exists()
    ),
    _SERVICE_FILE.parents[min(2, len(_SERVICE_FILE.parents) - 1)],
)


class FaceEmbeddingError(RuntimeError):
    pass


@dataclass
class FaceEmbeddingResult:
    vector: list[float]
    model_name: str
    backend_name: str
    metadata: dict[str, object]


class FaceEmbeddingBackend:
    backend_name = "hash"

    def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
        raise NotImplementedError

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
        """Extract every usable face while decoding/analyzing the image only once."""
        return [self.extract_embedding(image_bytes)]


class HashFaceEmbeddingBackend(FaceEmbeddingBackend):
    backend_name = "hash"

    def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
        digest = hashlib.sha256(image_bytes).hexdigest()
        values: list[float] = []
        for index in range(settings.recognition_embedding_dimensions):
            chunk = digest[(index * 4) % len(digest) : ((index * 4) % len(digest)) + 4]
            if len(chunk) < 4:
                chunk = (chunk + digest)[:4]
            value = int(chunk, 16) / 65535
            values.append(round((value * 2) - 1, 6))
        return FaceEmbeddingResult(
            vector=values,
            model_name=settings.recognition_embedding_model,
            backend_name=self.backend_name,
            metadata={
                "backend": self.backend_name,
                "deterministic": True,
                "dimensions": len(values),
            },
        )


class InsightFaceEmbeddingBackend(FaceEmbeddingBackend):
    backend_name = "insightface"

    def __init__(self) -> None:
        insightface_home = PROJECT_ROOT / "storage" / "insightface"
        insightface_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("INSIGHTFACE_HOME", str(insightface_home))

        try:
            import numpy as np
            from PIL import Image
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FaceEmbeddingError(
                "InsightFace dependencies are not installed. Install the recognition extras first."
            ) from exc

        self._np = np
        self._image = Image
        self._app = FaceAnalysis(
            name=settings.recognition_insightface_model,
            root=str(insightface_home),
            allowed_modules=["detection", "recognition"],
            providers=settings.recognition_insightface_providers,
        )
        self._app.prepare(
            ctx_id=settings.recognition_insightface_ctx_id,
            det_size=settings.recognition_insightface_det_size,
        )

    def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
        results = self.extract_embeddings(image_bytes)
        if not results:
            raise FaceEmbeddingError("No detectable face was found in the provided image.")
        return results[0]

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
        image = self._image.open(BytesIO(image_bytes)).convert("RGB")
        # InsightFace's FaceAnalysis uses OpenCV conventions and expects BGR.
        image_array = self._np.array(image)[:, :, ::-1]
        faces = self._app.get(image_array)
        if not faces:
            raise FaceEmbeddingError("No detectable face was found in the provided image.")

        ranked_faces = sorted(
            faces,
            key=lambda candidate: (
                float(getattr(candidate, "det_score", 0.0)),
                self._face_area(getattr(candidate, "bbox", None)),
            ),
            reverse=True,
        )
        results: list[FaceEmbeddingResult] = []
        for face_index, face in enumerate(ranked_faces):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            vector = [round(float(value), 6) for value in embedding.tolist()]
            bbox = getattr(face, "bbox", None)
            results.append(
                FaceEmbeddingResult(
                    vector=vector,
                    model_name=f"insightface-{settings.recognition_insightface_model}",
                    backend_name=self.backend_name,
                    metadata={
                        "backend": self.backend_name,
                        "deterministic": False,
                        "dimensions": len(vector),
                        "det_score": round(float(getattr(face, "det_score", 0.0)), 4),
                        "face_count": len(faces),
                        "face_index": face_index,
                        "bbox": (
                            [round(float(value), 2) for value in bbox.tolist()]
                            if bbox is not None
                            else None
                        ),
                    },
                )
            )

        if not results:
            raise FaceEmbeddingError("InsightFace did not return an embedding for any detected face.")
        return results

    @staticmethod
    def _face_area(bbox: object) -> float:
        if bbox is None or not hasattr(bbox, "tolist"):
            return 0.0
        values = bbox.tolist()
        if not isinstance(values, list) or len(values) != 4:
            return 0.0
        return max(0.0, float(values[2]) - float(values[0])) * max(
            0.0, float(values[3]) - float(values[1])
        )


def build_face_embedding_backend() -> FaceEmbeddingBackend:
    backend = settings.recognition_backend
    if backend == "hash":
        return HashFaceEmbeddingBackend()
    if backend == "insightface":
        return InsightFaceEmbeddingBackend()
    raise FaceEmbeddingError(f"Unsupported recognition backend: {settings.recognition_backend}")
