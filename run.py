from fastapi import FastAPI
from uvicorn import run
import config
from model import Model
from src.data_model import User, News, Response


app = FastAPI()
model = Model()


# @app.post('/login')
# def get_user_id(user: User):
#     return {
#         'user id': user
#     }


M = model.run_model()

@app.get('/recommend-news/{user_id}')
def recommend_news(user_id):
    # model.run_model(user_id)
    import kasifffffff
    kasifffffff.run_user_id = user_id
    # recommended_news = model.recommended_news
    recommended_news = next(M)
    return {
        'news' : recommended_news
    }


# @app.post('/response')
# def get_user_response(response: Response):
#     return {
#         'user_response': response
#     }

if __name__ == '__main__':
    run(app=app, host=config.HOST, port=config.PORT)

