from pydantic import EmailStr, BaseModel


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str
