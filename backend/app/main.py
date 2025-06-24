from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import database and router
from models.database import Base, engine
from routers import images

app = FastAPI(title="Quality Control Dashboard", version="1.0.0")

# CORS middleware - MUST be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://andedam.dev",
        "http://andedam.dev",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created/verified")
    print("✓ CORS middleware configured for andedam.dev")

# Include the upload router
app.include_router(images.router)

# Create uploads directory
os.makedirs("../uploads", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Quality Control Dashboard API", "version": "1.0.0"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "cors": "configured"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
