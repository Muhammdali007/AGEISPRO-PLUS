import argparse
import asyncio
from getpass import getpass

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.transactions import transaction_scope
from app.services.admin_recovery import AdminRecoveryError, reset_admin_password


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset or recreate the configured AegisPro administrator account."
    )
    parser.add_argument(
        "--email",
        default=settings.bootstrap_admin_email,
        help="Administrator email to recover. Defaults to BOOTSTRAP_ADMIN_EMAIL.",
    )
    parser.add_argument(
        "--password",
        help="New administrator password. If omitted, the command prompts securely.",
    )
    parser.add_argument(
        "--full-name",
        default="AegisPro Administrator",
        help="Full name to use when the administrator account must be recreated.",
    )
    return parser


async def run(email: str, password: str, full_name: str) -> str:
    async with AsyncSessionLocal() as session:
        async with transaction_scope(session):
            result = await reset_admin_password(
                session,
                email=email,
                password=password,
                full_name=full_name,
            )

    if result.created:
        return f"Created administrator account {result.email}."
    if result.reactivated:
        return f"Reset password and reactivated administrator account {result.email}."
    return f"Reset password for administrator account {result.email}."


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    password = args.password or getpass("New administrator password: ")
    try:
        message = asyncio.run(run(args.email, password, args.full_name))
    except AdminRecoveryError as exc:
        raise SystemExit(str(exc)) from exc
    print(message)


if __name__ == "__main__":
    main()
