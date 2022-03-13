from fastapi import FastAPI
from uvicorn import run
import config
from model import Model


app = FastAPI()
model = Model()


@app.get('/recommend-news/{user_id}')
def recommend_news(user_id: str) -> dict:
    recommended_news = model.recommend_news(user_id)
    return {
        'news' : recommended_news
    }


@app.get('/response/{user_response}')
def get_user_response(user_response: int) -> None:
    model.get_user_response(user_response)


if __name__ == '__main__':
    run(app=app, host=config.HOST, port=config.PORT)

