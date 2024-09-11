from fastapi import FastAPI
import uvicorn
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2


app = FastAPI()



@app.get("/")
def read_root():
   return {"message":"Action Recognition with VIVIT transformer model and fastapi"}


@app.get("/cctv", response_class=HTMLResponse)
def cctv_page():
    return """
    <html>
        <body>
            <h1>CCTV Live Feed</h1>
            <img src="/video_feed" width="640" height="480">
            <form action="/apply_ai" method="post">
                <button type="submit">Apply AI</button>
            </form>
        </body>
    </html>
    """

def gen_frames():
    camera = cv2.VideoCapture("rtsp://192.168.0.241/stream1")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
   uvicorn.run(app, host = "127.0.0.1",port = 8000)