from sqlalchemy import Boolean,Column,ForeignKey,Integer,String
from database import Base
from sqlalchemy.orm import relationship

class students(Base):
    __tablename__ = 'final_yr_students'

    reg_no = Column(Integer,primary_key=True,index=True)
    name = Column(String(50),nullable=False)
    branch = Column(String(50),nullable=False)
    email = Column(String(100),nullable=False)
    placed = Column(Boolean)

    placements = relationship("placement", back_populates="student", cascade="all, delete")

class placement(Base):
    __tablename__ = 'placement_info'

    placement_id = Column(Integer,primary_key=True,index=True)
    student_reg_no = Column(Integer,ForeignKey(students.reg_no),index=True)
    company = Column(String(100),index=True)
    package = Column(Integer)

    student = relationship("students", back_populates="placements")

