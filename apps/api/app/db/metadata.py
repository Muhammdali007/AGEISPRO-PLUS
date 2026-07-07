from app.db.base import Base

# Import models so Alembic and bootstrap metadata see the full schema.
from app.models.alert import Alert  # noqa: F401
from app.models.audit_log import AuditLog  # noqa: F401
from app.models.camera import Camera  # noqa: F401
from app.models.incident import Incident  # noqa: F401
from app.models.person import Person  # noqa: F401
from app.models.person_face_embedding import PersonFaceEmbedding  # noqa: F401
from app.models.user import User  # noqa: F401

__all__ = ["Base"]
