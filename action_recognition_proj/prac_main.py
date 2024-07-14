from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def get_post():
    return ("successful")