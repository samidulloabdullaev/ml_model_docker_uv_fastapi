import fastapi
import uvicorn

app = fastapi.FastAPI(title="Ping App")


@app.get("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
