from pydantic import BaseModel, StrictStr


class User(BaseModel):
    user_id: StrictStr

class Response(BaseModel):
    user_response: StrictStr

class News(BaseModel):
    news_id : StrictStr
    title: StrictStr
    abstarct: StrictStr
