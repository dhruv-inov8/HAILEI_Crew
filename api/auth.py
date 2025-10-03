"""
HAILEI Authentication and Authorization

Simple JWT-based authentication for API access control.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os


# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hailei-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security scheme
security = HTTPBearer(auto_error=False)

# Simple user database (in production, use proper database)
USERS_DB = {
    "admin": {
        "user_id": "admin_001",
        "username": "admin",
        "email": "admin@hailei.ai",
        "hashed_password": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        "role": "admin",
        "permissions": ["read", "write", "admin"]
    },
    "user": {
        "user_id": "user_001", 
        "username": "user",
        "email": "user@hailei.ai",
        "hashed_password": bcrypt.hashpw("user123".encode(), bcrypt.gensalt()).decode(),
        "role": "user",
        "permissions": ["read", "write"]
    },
    "demo": {
        "user_id": "demo_001",
        "username": "demo",
        "email": "demo@hailei.ai", 
        "hashed_password": bcrypt.hashpw("demo123".encode(), bcrypt.gensalt()).decode(),
        "role": "demo",
        "permissions": ["read"]
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = USERS_DB.get(username)
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    
    # Allow unauthenticated access for demo purposes
    # In production, you would enforce authentication
    if not credentials:
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "role": "user",  # Give user role for demo
            "permissions": ["read", "write"]
        }
    
    # Verify token
    payload = verify_token(credentials.credentials)
    if not payload:
        # In demo mode, return anonymous user instead of failing
        return {
            "user_id": "anonymous",
            "username": "anonymous", 
            "role": "user",
            "permissions": ["read", "write"]
        }
    
    # Get user from payload
    username = payload.get("sub")
    if not username:
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "role": "user", 
            "permissions": ["read", "write"]
        }
    
    user = USERS_DB.get(username)
    if not user:
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "role": "user",
            "permissions": ["read", "write"]
        }
    
    return user


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker


def require_role(role: str):
    """Decorator to require specific role"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        if current_user.get("role") != role and current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required"
            )
        return current_user
    
    return role_checker


# Authentication endpoints
from fastapi import APIRouter
from api.models import TokenRequest, TokenResponse, UserInfo

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/login", response_model=TokenResponse)
async def login(request: TokenRequest):
    """Authenticate user and return access token"""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@auth_router.get("/me", response_model=UserInfo)
async def get_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return UserInfo(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user.get("email"),
        role=current_user["role"],
        permissions=current_user["permissions"]
    )


@auth_router.post("/refresh")
async def refresh_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Refresh access token"""
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user["username"], "role": current_user["role"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


# Demo credentials endpoint
@auth_router.get("/demo-credentials")
async def get_demo_credentials():
    """Get demo credentials for testing"""
    return {
        "message": "Demo credentials for testing",
        "credentials": [
            {
                "username": "admin",
                "password": "admin123",
                "role": "admin",
                "permissions": ["read", "write", "admin"]
            },
            {
                "username": "user", 
                "password": "user123",
                "role": "user",
                "permissions": ["read", "write"]
            },
            {
                "username": "demo",
                "password": "demo123", 
                "role": "demo",
                "permissions": ["read"]
            }
        ],
        "note": "These are demo credentials. Change in production!"
    }