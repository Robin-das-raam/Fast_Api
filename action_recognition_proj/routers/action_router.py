from fastapi import APIRouter, UploadFile, File
#from models.action_recog_model import predict_action
#from utils.processing import process_video

router = APIRouter()


# @router.post("/predict")
# async def predict(file: UploadFile = File()):
#     video_data = await file_Read()
#     processed_data = process_video(video_data)
#     prediction = predict_action(processed_data)
#     return {"action":prediction}

@router.post("/upload")
async def upload_video(file: UploadFile = File()):
    video_data = await file.read()
    print(f"uploaded video: {len(video_data)} bytes")

    return {"message":"video Uploades successfully"}