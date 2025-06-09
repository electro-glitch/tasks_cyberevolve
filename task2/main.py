from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import relationship

secret_key = 'abcd@1234!'
algorithm = 'HS256'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')


def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail='invalid token')
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail='invalid token')


app = FastAPI()
models.Base.metadata.create_all(bind=engine)  # creates all the tables, columns in postgresql


@app.post('/token/')
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not form_data.username:
        raise HTTPException(status_code=400, detail='username required!')
    if form_data.username != 'admin' or form_data.password != 'password':
        raise HTTPException(status_code=200, detail='unauthorized access attempted')

    payload = {"sub": form_data.username}
    token_jwt = jwt.encode(payload, secret_key, algorithm=algorithm)

    return {"access_token": token_jwt, "token_type": "bearer"}


@app.get('/')
async def index(username: str = Depends(verify_token)):
    return {"username": username}


# BASE MODELS

class studentsBase(BaseModel):
    reg_no: int
    name: str
    branch: str
    email: str
    placed: bool


# New model for adding student + placement if applicable
class studentsCreate(BaseModel):
    reg_no: int
    name: str
    branch: str
    email: str
    placed: bool
    company: str | None = None
    package: int | None = None


class placementBase(BaseModel):
    placement_id: int
    student_reg_no: int
    company: str
    package: int


# DATABASE DEPENDENCY

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


# ENDPOINTS

@app.post("/students/add_students")
async def create_students(student: studentsCreate, db: db_dependency, username: str = Depends(verify_token)):
    db_student = models.students(
        reg_no=student.reg_no,
        name=student.name,
        branch=student.branch,
        email=student.email,
        placed=student.placed
    )
    db.add(db_student)
    db.commit()
    db.refresh(db_student)

    # If placed==True → insert placement
    if student.placed:
        if student.company is None or student.package is None:
            raise HTTPException(status_code=400, detail="Company and package required when placed=True")

        db_placement = models.placement(
            placement_id=None,  # assuming placement_id is autoincrement primary key
            student_reg_no=student.reg_no,
            company=student.company,
            package=student.package
        )
        db.add(db_placement)
        db.commit()
        db.refresh(db_placement)

    return {"detail": "Student (and placement if applicable) added successfully"}


@app.get("/students/{reg_no}")
async def read_question(reg_no: int, db: db_dependency, username: str = Depends(verify_token)):
    result = db.query(models.students).filter(models.students.reg_no == reg_no).first()

    if not result:
        raise HTTPException(status_code=404, detail='student not found')
    return result


@app.post("/placements/create_data")
async def add_placement_info(placement: placementBase, db: db_dependency, username: str = Depends(verify_token)):
    student = db.query(models.students).filter(models.students.reg_no == placement.student_reg_no).first()

    if not student:
        raise HTTPException(status_code=404, detail='Student does not exist')

    if not student.placed:
        raise HTTPException(status_code=400, detail='Student not marked as placed')

    db_placement = models.placement(
        placement_id=placement.placement_id,
        student_reg_no=placement.student_reg_no,
        company=placement.company,
        package=placement.package
    )

    db.add(db_placement)
    db.commit()
    db.refresh(db_placement)

    return db_placement


@app.get("/placements/get_placement_info")
async def get_placement_data(placement_id: int, db: db_dependency, username: str = Depends(verify_token)):
    data = db.query(models.placement).filter(placement_id == models.placement.placement_id).first()

    if not data:
        raise HTTPException(status_code=404, detail='Placement record not found')
    return data


@app.put('/placements/update_info/{student_reg_no}')
async def update_placement_info(student_reg_no: int, new_company: str, new_package: int, db: db_dependency,
                                 username: str = Depends(verify_token)):
    record = db.query(models.placement).filter(student_reg_no == models.placement.student_reg_no).first()

    if not record:
        raise HTTPException(status_code=404, detail='Placement record not found')

    record.package = new_package
    record.company = new_company

    db.commit()
    db.refresh(record)
    return record


@app.put('/students/update_info/{reg_no}')
async def update_student_info(reg_no: int, name: str, branch: str, email: str, placed: bool, db: db_dependency,
                              username: str = Depends(verify_token)):
    record = db.query(models.students).filter(reg_no == models.students.reg_no).first()

    if not record:
        raise HTTPException(status_code=404, detail='Student not found')

    old_rno = reg_no
    record.reg_no = reg_no
    record.name = name
    record.branch = branch
    record.email = email
    record.placed = placed

    # If placed changed to False → delete placement record if exists
    if not placed:
        del_r = db.query(models.placement).filter(
            (models.placement.student_reg_no == old_rno) | (models.placement.student_reg_no == record.reg_no)).first()

        if del_r:
            db.delete(del_r)
            db.commit()

    db.commit()
    db.refresh(record)
    return record


@app.delete('/students/delete_student/{regno}')
async def del_student_info(regno: int, db: db_dependency, username: str = Depends(verify_token)):
    del_r = db.query(models.students).filter(models.students.reg_no == regno).first()

    if not del_r:
        raise HTTPException(status_code=404, detail='Student not found')

    db.delete(del_r)
    db.commit()

    return {'detail': 'Student deleted successfully'}


@app.delete('/placement/delete_placement/{id}')
async def del_placement_info(id: int, db: db_dependency, username: str = Depends(verify_token)):
    del_r = db.query(models.placement).filter(models.placement.placement_id == id).first()

    if not del_r:
        raise HTTPException(status_code=404, detail='Placement record not found')

    db.delete(del_r)
    db.commit()

    return {'detail': 'Placement record deleted successfully'}

@app.get("/students/all")
async def get_all_students(db: db_dependency, username: str = Depends(verify_token)):
    students = db.query(models.students).all()

    if not students:
        raise HTTPException(status_code=404, detail='No students found')

    return students

@app.get("/placements/all")
async def get_all_placements(db: db_dependency, username: str = Depends(verify_token)):
    placements = db.query(models.placement).all()

    if not placements:
        raise HTTPException(status_code=404, detail='No placements found')

    return placements