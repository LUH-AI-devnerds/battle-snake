from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def model():
    return {"action": "move"}

@app.get("/info")
def infofunction():
    return {"message": "everything is working"} 