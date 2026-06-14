import os
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

from docmind_api.models import Tenant

load_dotenv()
dsn = os.environ.get("DATABASE_URL")

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(dsn)


async def close_pool() -> None:
    if not _pool:
        raise RuntimeError("_pool not initialized")
    await _pool.close()


def get_pool() -> asyncpg.Pool:
    if not _pool:
        raise RuntimeError("_pool not initialized")
    return _pool


async def get_tenant_by_id(tenant_id: UUID) -> Tenant | None:
    pool = get_pool()
    record = await pool.fetchrow(
        "SELECT id, name, slug, created_at FROM tenants WHERE id = $1", tenant_id
    )
    if not record:
        return None
    return Tenant(
        id=record["id"],
        name=record["name"],
        slug=record["slug"],
        created_at=record["created_at"],
    )


async def get_tenant_by_slug(slug: str) -> tuple[Tenant, str] | None:
    pool = get_pool()
    record = await pool.fetchrow(
        "SELECT id, name, slug, created_at, hashed_password"
        " FROM tenants WHERE slug = $1",
        slug,
    )
    if not record:
        return None
    return (
        Tenant(
            id=record["id"],
            name=record["name"],
            slug=record["slug"],
            created_at=record["created_at"],
        ),
        record["hashed_password"],
    )


async def create_tenant(name: str, slug: str, hashed_password: str) -> None:
    pool = get_pool()
    try:
        await pool.execute(
            "INSERT INTO tenants (name, slug, hashed_password) VALUES ($1, $2, $3)",
            name,
            slug,
            hashed_password,
        )
    except asyncpg.UniqueViolationError:
        raise ValueError("tenant already exists")
