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
        "dotenv": "python-dotenv",
        "boto3": "boto3"
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

def check_aws_credentials():
    """Verify AWS credentials are valid at all, via STS"""
    print("\n" + "="*60)
    print("5️⃣  AWS Credentials Check (STS)")
    print("="*60)

    try:
        import boto3
        region = os.getenv("AWS_REGION", "ap-south-1")

        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            print("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing in .env")
            return False

        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()

        print("✅ AWS credentials valid")
        print(f"✅ Account ID: {identity['Account']}")
        print(f"✅ IAM ARN   : {identity['Arn']}")
        return True

    except Exception as e:
        print(f"❌ AWS credential check failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")
        print("  2. Confirm the access key hasn't been deleted/deactivated in IAM")
        return False

def check_ses_connection():
    """Verify SES is reachable and check verified identity status"""
    print("\n" + "="*60)
    print("6️⃣  SES (Email) Check")
    print("="*60)

    try:
        import boto3
        region = os.getenv("AWS_REGION", "ap-south-1")
        sender = os.getenv("SES_SENDER_EMAIL")
        receiver = os.getenv("SES_RECEIVER_EMAIL")

        if not sender or not receiver:
            print("❌ SES_SENDER_EMAIL / SES_RECEIVER_EMAIL missing in .env")
            return False

        ses = boto3.client("ses", region_name=region)

        # get_send_quota confirms the client can reach SES at all
        quota = ses.get_send_quota()
        print(f"✅ SES reachable — daily quota: {quota['Max24HourSend']}, "
              f"sent in last 24h: {quota['SentLast24Hours']}")

        # Check verified identities
        verified = ses.list_verified_email_addresses()
        verified_emails = verified.get("VerifiedEmailAddresses", [])

        if sender in verified_emails:
            print(f"✅ Sender verified: {sender}")
        else:
            print(f"❌ Sender NOT verified: {sender} — check SES console inbox for verification link")

        if receiver in verified_emails:
            print(f"✅ Receiver verified: {receiver}")
        else:
            print(f"❌ Receiver NOT verified: {receiver} — required in SES sandbox mode")

        return sender in verified_emails and receiver in verified_emails

    except Exception as e:
        print(f"❌ SES check failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Confirm IAM policy allows ses:SendEmail / ses:ListVerifiedEmailAddresses")
        print("  2. Confirm AWS_REGION matches where you set up SES")
        return False

def check_sqs_connection():
    """Verify SQS queue is reachable"""
    print("\n" + "="*60)
    print("7️⃣  SQS (Queue) Check")
    print("="*60)

    try:
        import boto3
        region = os.getenv("AWS_REGION", "ap-south-1")
        queue_url = os.getenv("SQS_QUEUE_URL")

        if not queue_url:
            print("❌ SQS_QUEUE_URL missing in .env")
            return False

        sqs = boto3.client("sqs", region_name=region)
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages", "QueueArn"]
        )

        print(f"✅ SQS queue reachable: {queue_url}")
        print(f"✅ Queue ARN: {attrs['Attributes'].get('QueueArn')}")
        print(f"✅ Messages currently in queue: {attrs['Attributes'].get('ApproximateNumberOfMessages')}")
        return True

    except Exception as e:
        print(f"❌ SQS check failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Confirm SQS_QUEUE_URL is correct in .env")
        print("  2. Confirm IAM policy allows sqs:GetQueueAttributes")
        return False

def check_cloudwatch_connection():
    """Verify CloudWatch is reachable by sending a harmless test metric"""
    print("\n" + "="*60)
    print("8️⃣  CloudWatch Check")
    print("="*60)

    try:
        import boto3
        region = os.getenv("AWS_REGION", "ap-south-1")
        cloudwatch = boto3.client("cloudwatch", region_name=region)

        cloudwatch.put_metric_data(
            Namespace="SmartTicketAI",
            MetricData=[{
                "MetricName": "DebugScriptPing",
                "Value": 1,
                "Unit": "Count"
            }]
        )
        print("✅ CloudWatch reachable — test metric 'DebugScriptPing' sent")
        print("   Check: CloudWatch → Metrics → SmartTicketAI namespace")
        return True

    except Exception as e:
        print(f"❌ CloudWatch check failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Confirm IAM policy allows cloudwatch:PutMetricData")
        return False

def check_s3_connection():
    """Verify S3 bucket is reachable"""
    print("\n" + "="*60)
    print("9️⃣  S3 (Model Storage) Check")
    print("="*60)

    try:
        import boto3
        region = os.getenv("AWS_REGION", "ap-south-1")
        bucket = os.getenv("S3_BUCKET_NAME")

        if not bucket:
            print("❌ S3_BUCKET_NAME missing in .env")
            return False

        s3 = boto3.client("s3", region_name=region)
        response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)

        print(f"✅ S3 bucket reachable: {bucket}")
        if "Contents" in response:
            files = [obj["Key"] for obj in response["Contents"]]
            print(f"✅ Sample files found: {', '.join(files)}")
        else:
            print("⚠️  Bucket is reachable but empty")
        return True

    except Exception as e:
        print(f"❌ S3 check failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Confirm S3_BUCKET_NAME is correct in .env")
        print("  2. Confirm IAM policy allows s3:ListBucket / s3:GetObject on this bucket")
        return False

def test_application():
    """Test if the FastAPI application can start"""
    print("\n" + "="*60)
    print("🔟 Application Check")
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
        ("AWS Credentials", check_aws_credentials),
        ("SES", check_ses_connection),
        ("SQS", check_sqs_connection),
        ("CloudWatch", check_cloudwatch_connection),
        ("S3", check_s3_connection),
        ("Application", test_application),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {str(e)}")
            results[name] = False
    
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