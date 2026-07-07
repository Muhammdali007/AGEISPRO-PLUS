from uuid import UUID

from pydantic import BaseModel, Field

from app.models.user import UserRole


class UserRead(BaseModel):
    id: UUID
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    full_name: str
    role: UserRole
    is_active: bool

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    full_name: str = Field(min_length=1, max_length=160)
    role: UserRole = UserRole.viewer
    password: str = Field(min_length=8, max_length=128)
    is_active: bool = True


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    full_name: str | None = Field(default=None, min_length=1, max_length=160)
    role: UserRole | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)
    is_active: bool | None = None


class UserSelfUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=160)
    password: str | None = Field(default=None, min_length=8, max_length=128)
