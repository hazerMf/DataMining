import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn
    from src.main import app
    from src.config import Config

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        reload=False 
    )
