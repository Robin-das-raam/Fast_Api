from fastapi import FastAPI
import uvicorn
from routers.action_router import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def read_root():
   return {"message":"Welcome to the action recognition API"}