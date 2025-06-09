#used by SQL Alchemy to create the tables in the database

# Import SQLAlchemy classes used to define table columns:
# - Column: Base class to define columns in ORM models
# - Integer: For integer/whole number columns
# - String: For text/varchar columns
# - Boolean: For true/false columns
from sqlalchemy import Boolean, Column, Integer, String

from mysql.database import Base 

class my_Table(Base):
    __tablename__ = 'my_table'

    reg_no = Column(Integer,primary_key=True,index=True)
    name = Column(String(70),nullable=False)
    branch = Column(String(50),nullable=False)
    email = Column(String(100),unique=True,nullable=False)