import os
from datetime import datetime, timedelta
from uuid import UUID

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from docmind_api.db import get_tenant_by_id
from docmind_api.models import Tenant

load_dotenv()
secret = os.environ.get("SECRET_KEY", "")
if not secret:
    raise RuntimeError("SECRET_KEY environment variable is not set")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(tenant_id: UUID) -> str:
    payload = {"tenant_id": str(tenant_id), "exp": datetime.now() + timedelta(hours=3)}
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> UUID:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
    return UUID(payload["tenant_id"])


async def get_current_tenant(token: str = Depends(oauth2_scheme)) -> Tenant:
    try:
        tenant_id = decode_access_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await get_tenant_by_id(tenant_id)
    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return result
