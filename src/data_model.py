from pydantic import BaseModel, StrictStr


class User(BaseModel):
    user_id: StrictStr

class Response(BaseModel):
    user_response: int

class News(BaseModel):
    news_id : StrictStr
    title: StrictStr
    abstract: StrictStr
