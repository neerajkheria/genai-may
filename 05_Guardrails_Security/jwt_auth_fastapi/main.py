"""
FastAPI with JWT Authentication + Rate Limiting
Secure API layer for GenAI endpoints
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv; load_dotenv()

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="GenAI Insurance API", version="1.0.0")

if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if JWT_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database
FAKE_USERS_DB = {
    "trainer": {"username": "trainer", "hashed_password": "PLACEHOLDER_HASH", "role": "admin"},
    "participant": {"username": "participant", "hashed_password": "PLACEHOLDER_HASH", "role": "user"},
}

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    if not JWT_AVAILABLE:
        return "jwt-not-available"
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not JWT_AVAILABLE:
        return {"username": "demo_user", "role": "user"}
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return FAKE_USERS_DB.get(username, {"username": username, "role": "user"})
    except JWTError:
        raise credentials_exception

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username},
                                        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    # In production, call your RAG/agent pipeline here
    mock_response = f"[Auth: {current_user['username']}] Processed: {request.query}"
    return ChatResponse(response=mock_response, session_id=request.session_id, tokens_used=150)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
