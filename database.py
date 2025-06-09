#Contains all the connection strings for thr database we will set up


from sqlalchemy import create_engine #setting up a configuration to connect to the DB
from sqlalchemy.orm import sessionmaker #creates class to set up sessions
from sqlalchemy.ext.declarative import declarative_base #creates a base class that orm inherits from

URL_DATABASE = 'mysql+pymysql://root:tanay@localhost:3306/my_db'

engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

