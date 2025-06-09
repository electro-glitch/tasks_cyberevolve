# FastAPI core classes and utilities:
from fastapi import FastAPI, HTTPException, Depends, status

# Pydantic for data validation and serialization
from pydantic import BaseModel

# Annotated allows adding metadata or dependencies to function parameters
from typing import Annotated    

# Import SQLAlchemy models
import mysql.models as models

# Import the database engine and session factory
from mysql.database import engine, SessionLocal

# SQLAlchemy ORM session
from sqlalchemy.orm import Session

# Create FastAPI app
app = FastAPI()

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Pydantic model for student
class MyTableBase(BaseModel):
    reg_no: int
    name: str
    branch: str
    email: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Annotated DB dependency
db_dependency = Annotated[Session, Depends(get_db)]

# CREATE a new student
@app.post("/students/", status_code=status.HTTP_201_CREATED)
async def create_student(student: MyTableBase, db: db_dependency):
    db_student = models.my_Table(**student.dict())
    db.add(db_student)
    db.commit()

# READ ALL students
@app.get("/students/", status_code=status.HTTP_200_OK)
async def get_all_students(db: db_dependency):
    students = db.query(models.my_Table).all()
    return students

# READ ONE student by reg_no
@app.get("/students/{reg_no}", status_code=status.HTTP_200_OK)
async def get_student(reg_no: int, db: db_dependency):
    student = db.query(models.my_Table).filter(models.my_Table.reg_no == reg_no).first()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

# UPDATE a student by reg_no
@app.put("/students/{reg_no}", status_code=status.HTTP_200_OK)
async def update_student(reg_no: int, updated_student: MyTableBase, db: db_dependency):
    db_student = db.query(models.my_Table).filter(models.my_Table.reg_no == reg_no).first()
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Update fields
    db_student.name = updated_student.name
    db_student.branch = updated_student.branch
    db_student.email = updated_student.email
    
    db.commit()
    db.refresh(db_student)  # Optional: return the updated object
    return db_student

# DELETE a student by reg_no
@app.delete("/students/{reg_no}", status_code=status.HTTP_200_OK)
async def delete_student(reg_no: int, db: db_dependency):
    db_student = db.query(models.my_Table).filter(models.my_Table.reg_no == reg_no).first()
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.delete(db_student)
    db.commit()
    return {"detail": "Student deleted successfully"}
