import logging
import os
from datetime import datetime

# Step 1: Create logs folder path
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Step 2: Create log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Step 3: Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Step 4: Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [Line:%(lineno)d] [%(name)s] [%(levelname)s] - %(message)s",
    level=logging.INFO,
)

