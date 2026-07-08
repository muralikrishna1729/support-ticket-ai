"""
Database and Application Debug Script
Use this to diagnose and test your PostgreSQL setup
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text

def check_environment():
    """Check if .env file exists and has required variables"""
    print("\n" + "="*60)
    print("1️⃣  Environment Check")
    print("="*60)
    
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        return False
    
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("❌ DATABASE_URL not set in .env")
        return False
    
    print(f"✅ .env file exists")
    print(f"✅ DATABASE_URL configured: {db_url.split('@')[0]}@***")
    return True

def check_imports():
    """Check if all required packages are installed"""
    print("\n" + "="*60)
    print("2️⃣  Dependencies Check")
    print("="*60)
    
    required_packages = {
        "sqlalchemy": "SQLAlchemy",
        "psycopg2": "psycopg2-binary",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "dotenv": "python-dotenv"
    }
    
    all_ok = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed - run: pip install {name}")
            all_ok = False
    
    return all_ok

def check_database_connection():
    """Test database connection"""
    print("\n" + "="*60)
    print("3️⃣  Database Connection Check")
    print("="*60)
    
    try:
        from src.db.database import engine
        with engine.connect() as conn:
            print("✅ Database connection successful!")
            
            # Get database info
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()
            print(f"✅ PostgreSQL Version: {version[0].split(',')[0]}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check if PostgreSQL is running")
        print("  2. Verify DATABASE_URL in .env file")
        print("  3. Check if database and user exist")
        return False

def check_database_tables():
    """Check if database tables exist"""
    print("\n" + "="*60)
    print("4️⃣  Database Tables Check")
    print("="*60)
    
    try:
        from src.db.database import Base, engine
        from src.db import models
        
        # Get table names
        inspector_sql = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public';
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(inspector_sql))
            tables = [row[0] for row in result.fetchall()]
            
            if tables:
                print(f"✅ Found tables: {', '.join(tables)}")
                if 'tickets' in tables:
                    print("✅ 'tickets' table exists")
                else:
                    print("❌ 'tickets' table not found - creating tables...")
                    Base.metadata.create_all(bind=engine)
                    print("✅ Tables created!")
            else:
                print("❌ No tables found - creating tables...")
                Base.metadata.create_all(bind=engine)
                print("✅ Tables created!")
        return True
    except Exception as e:
        print(f"❌ Error checking tables: {str(e)}")
        return False

def test_application():
    """Test if the FastAPI application can start"""
    print("\n" + "="*60)
    print("5️⃣  Application Check")
    print("="*60)
    
    try:
        from src.app import app
        print("✅ FastAPI application imported successfully")
        print(f"✅ Application title: {app.title}")
        print(f"✅ Application version: {app.version}")
        return True
    except Exception as e:
        print(f"❌ Error loading application: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 Support Ticket AI - Debug & Setup Verification")
    print("="*60)
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_imports),
        ("Database Connection", check_database_connection),
        ("Database Tables", check_database_tables),
        ("Application", test_application),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {str(e)}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\nRun the application with:")
        print("  uvicorn src.app:app --reload")
        print("\nAPI Documentation:")
        print("  http://localhost:8000/docs")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
