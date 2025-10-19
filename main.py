# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import motor.motor_asyncio
import google.generativeai as genai
from bson.objectid import ObjectId

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in .env or environment before starting")
genai.configure(api_key=GOOGLE_API_KEY)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "appdb")  # set your DB name in env
if not MONGODB_URI:
    raise RuntimeError("Set MONGODB_URI in .env or environment before starting")

# Motor client (async)
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB]
profiles_coll = db["user_profile"]
diary_coll = db["diary"]

app = FastAPI(title="AI Mental Health Companion - Backend (Mongo)")

origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class UserProfile(BaseModel):
    nickname: Optional[str] = ""
    age: Optional[str] = "select"
    occupation: Optional[str] = ""
    medical_conditions: Optional[str] = "None"

class ChatRequest(BaseModel):
    message: str
    profile: Optional[UserProfile] = None
    mood: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class DiaryIn(BaseModel):
    date: Optional[str] = None
    text: str

class DiaryOut(BaseModel):
    id: str
    date: str
    text: str

# Helper: get or create single profile (single-user simple model)
async def get_or_create_profile():
    p = await profiles_coll.find_one({})
    if not p:
        doc = {
            "nickname": "",
            "age": "select",
            "occupation": "",
            "medical_conditions": "None",
            "updated_at": datetime.utcnow()
        }
        res = await profiles_coll.insert_one(doc)
        doc["_id"] = res.inserted_id
        return doc
    return p

@app.get("/profile", response_model=UserProfile)
async def read_profile():
    p = await get_or_create_profile()
    return UserProfile(
        nickname=p.get("nickname", ""),
        age=p.get("age", "select"),
        occupation=p.get("occupation", ""),
        medical_conditions=p.get("medical_conditions", "None")
    )

@app.put("/profile", response_model=UserProfile)
async def update_profile(profile: UserProfile):
    p = await get_or_create_profile()
    update = {
        "$set": {
            "nickname": profile.nickname or p.get("nickname", ""),
            "age": profile.age or p.get("age", "select"),
            "occupation": profile.occupation or p.get("occupation", ""),
            "medical_conditions": profile.medical_conditions or p.get("medical_conditions", "None"),
            "updated_at": datetime.utcnow()
        }
    }
    await profiles_coll.update_one({"_id": p["_id"]}, update)
    new_p = await profiles_coll.find_one({"_id": p["_id"]})
    return UserProfile(
        nickname=new_p.get("nickname", ""),
        age=new_p.get("age", "select"),
        occupation=new_p.get("occupation", ""),
        medical_conditions=new_p.get("medical_conditions", "None")
    )

@app.post("/diary", response_model=DiaryOut)
async def create_diary(entry: DiaryIn):
    date_str = entry.date or datetime.utcnow().strftime("%Y-%m-%d")
    # If you want one entry per day and append existing:
    existing = await diary_coll.find_one({"date": date_str})
    if existing:
        new_text = existing.get("text", "") + "\n\n---\n\n" + entry.text
        await diary_coll.update_one({"_id": existing["_id"]}, {"$set": {"text": new_text}})
        existing = await diary_coll.find_one({"_id": existing["_id"]})
        return DiaryOut(id=str(existing["_id"]), date=existing["date"], text=existing["text"])
    doc = {"date": date_str, "text": entry.text, "created_at": datetime.utcnow()}
    res = await diary_coll.insert_one(doc)
    doc["_id"] = res.inserted_id
    return DiaryOut(id=str(doc["_id"]), date=doc["date"], text=doc["text"])

@app.get("/diary", response_model=List[DiaryOut])
async def list_diary(limit: int = 50):
    cursor = diary_coll.find({}, sort=[("date", -1)]).limit(limit)
    items = []
    async for d in cursor:
        items.append(DiaryOut(id=str(d["_id"]), date=d.get("date", ""), text=d.get("text", "")))
    return items

@app.delete("/diary/{entry_id}")
async def delete_diary(entry_id: str):
    try:
        oid = ObjectId(entry_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid entry id")
    res = await diary_coll.delete_one({"_id": oid})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Diary entry not found")
    return {"ok": True}

# Gemini helper (same as before)
def get_gemini_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        content = [prompt]
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    profile = req.profile or UserProfile()
    prompt = f"""
Dear {profile.nickname or 'User'},
Mood: {req.mood or 'not provided'}
Age: {profile.age}
Occupation: {profile.occupation}
Medical conditions: {profile.medical_conditions}

User message: {req.message}

Instructions:
1) Reply empathetically, match mood.
2) Offer a CBT-inspired coping thought or affirmation.
3) Suggest a simple activity (mindful breathing, short stretch, grounding).
4) Start with a friendly greeting and end with a gentle disclaimer that you're an AI, not a substitute for a professional.
Format with headings and bullet points.
"""
    reply = get_gemini_response(prompt)
    return ChatResponse(reply=reply)
