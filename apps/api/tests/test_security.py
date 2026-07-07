from uuid import uuid4

from app.core.security import TokenType, create_token, decode_token, hash_password, verify_password


def test_password_hashing_roundtrip() -> None:
    password_hash = hash_password("ChangeMe123!")

    assert verify_password("ChangeMe123!", password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_access_token_roundtrip() -> None:
    user_id = uuid4()
    token = create_token(user_id, "administrator", TokenType.access)
    payload = decode_token(token, TokenType.access)

    assert payload["sub"] == str(user_id)
    assert payload["role"] == "administrator"
    assert payload["type"] == "access"

