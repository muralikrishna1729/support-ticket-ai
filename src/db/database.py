from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
from src.logger import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tickets.db")

# Connection parameters
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
pool_kwargs = {} if "sqlite" in DATABASE_URL else {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,  # Test connection before using
    "pool_recycle": 3600,  # Recycle connections after 1 hour
}

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        **pool_kwargs,
        echo=False  # Set to True for SQL debugging
    )
    
    # Test connection
    with engine.connect() as conn:
        logger.info("✅ Database connection successful")
        
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    raise

sessionLocal = sessionmaker(
    bind       = engine,
    autoflush  = False,
    autocommit = False
)

Base = declarative_base()