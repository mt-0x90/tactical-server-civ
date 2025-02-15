import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot import types
from rag import RagPipeline
from pathlib import Path
from sttHandler import SttPipeline
import json
import os

creds = json.load(open(os.environ['AGENTCONFIG']))
bot = AsyncTeleBot(creds["telegram"])
stt_handler = SttPipeline(config=creds, use_local=True)
rag = RagPipeline(use_ollama=True)

his_dir = Path("history")
scene_type = "civil"
current_msg = {}

def save_history(data):
    his_fpath = os.path.join(his_dir, "history.json")
    if os.path.exists(his_fpath):
        old = json.load(open(his_fpath))
        old.append(data)
        with open(his_fpath, 'w') as f:
            json.dump(old, f)
        f.close()
    else:
        with open(his_fpath, 'w') as f:
            json.dump([data], f)
        f.close()

def gen_markup():
    markup = types.InlineKeyboardMarkup()
    markup.row_width = 2
    like_button = types.InlineKeyboardButton(text="👍 Like", callback_data="like")
    dislike_button = types.InlineKeyboardButton(text="👎 Dislike", callback_data="dislike")
    markup.add(like_button, dislike_button)
    return markup

	
@bot.message_handler(content_types=['voice'])
async def new_user_audio(message: telebot.types.Message):
    global scene_type
    result_message = await bot.send_message(message.chat.id, '<i>Downloading your audio...</i>', parse_mode='HTML', disable_web_page_preview=True)
    file_path = await bot.get_file(message.voice.file_id)
    downloaded_file = await bot.download_file(file_path.file_path)
    audio_file = stt_handler.fileUtils.get_alt_fpath()
    with open(audio_file, "wb") as f:
        f.write(downloaded_file)
    user_text = stt_handler.get_text(audio_file, is_wav=True)
    print(user_text)
    answer, udp_signal, is_scene = rag.handle_user_text(user_text=user_text,voice_id="chris",scene_type=scene_type)
    save_history({"user_audio":audio_file, "model_prediction":user_text, "answer":answer, "is_scene": is_scene})
    with open(answer, "rb") as mp3_file:
        await bot.send_audio(message.chat.id, mp3_file, reply_markup=gen_markup())

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    text = 'Hi, I am Tactica!'
    await bot.reply_to(message, text)

@bot.message_handler(commands=['military'])
async def switch_scene_types(message):
    global scene_type
    scene_type = "military"
    await bot.reply_to(message, "Scene switched to military")

@bot.message_handler(commands=['civil'])
async def switch_scene_civi(message):
    global scene_type
    scene_type = "civil"
    await bot.reply_to(message, "Scene switched to civilian")

@bot.message_reaction_handler(func=lambda message: True)
async def get_reactions(message):
    bot.reply_to(message, f"You changed the reaction from {[r.emoji for r in message.old_reaction]} to {[r.emoji for r in message.new_reaction]}")

@bot.callback_query_handler(func=lambda call: True)
async def callback_query(call):
    global current_msg
    if call.data == "like":
        current_msg['feedback'] = "like"
        await bot.answer_callback_query(call.id, "😄")
    elif call.data == "dislike":
        current_msg['feedback'] = "dislike"
        await bot.answer_callback_query(call.id, "Thanks for the feedback 😉")
    stt_handler.fileUtils.save_feedback(current_msg)
    current_msg = {}

    

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def new_user_text(message):
    global current_msg
    answer, udp_signal, is_scene = rag.handle_user_text(user_text=message.text,voice_id="matilda", is_local=True)
    save_history({"user_audio":"na", "model_prediction":message.text, "answer":answer, "is_scene": is_scene})
    current_msg['user'] = message.text
    current_msg['tactica'] = answer
    current_msg['is_scene'] = is_scene
    if is_scene:
        with open(answer, "rb") as mp3_file:
            await bot.send_audio(message.chat.id, mp3_file, reply_markup=gen_markup())
    else:
        await bot.reply_to(message, answer, reply_markup=gen_markup())


import asyncio
asyncio.run(bot.polling())