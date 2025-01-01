from pathlib import Path
from openai import OpenAI
import os

def read_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_string_into_chunks(input_string, chunk_size=4096):
    return [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]

#os.environ["OPENAI_API_KEY"] = "<your key here>"

chat_string = read_file_to_string("./conversations/global_conversation.txt")
chunk_array = split_string_into_chunks(chat_string)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

client = OpenAI()
for i in range(len(chunk_array)):

    speech_file_path = Path(__file__).parent / f"global_conversation_{i}.mp3"
    print(f"working on {speech_file_path} out of {len(chunk_array)}")
    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=chunk_array[i],
    )
    response.stream_to_file(speech_file_path)