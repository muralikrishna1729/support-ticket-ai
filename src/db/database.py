from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base   
DATABASE_URL = "sqlite:///./tickets.db" 

engine = create_engine(DATABASE_URL,connect_args = {"check_same_thread":False})
 ## Setting this to False allows multiple threads (e.g., web server requests) to use the same database connection


sessionLocal = sessionmaker(bind =engine,autoflush = False,autocommit = False)

Base =declarative_base() 

# You’ll define models using Base.
# Each request gets a session via SessionLocal().
# Changes are saved manually with .commit()