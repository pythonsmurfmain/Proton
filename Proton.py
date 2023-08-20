import random
import json
import torch
import tkinter as tk
from pathlib import Path
from tkinter import Entry, Text, Button, Canvas, PhotoImage
from NeuralNetwork import NN
from Nltk import tokenize, bgfw
import pyttsx3

def speak(text, rate=150):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()
    
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"F:\G\build\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def load_data(file_path):
    data = torch.load(file_path)
    return data

def process_input(model, sentence, all_words, tags):
    sentence = tokenize(sentence)
    X = bgfw(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob

def generate_response(model, intents, sentence, all_words, tags):
    tag, prob = process_input(model, sentence, all_words, tags)

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return response
    else:
        return "I do not understand..."

def send_message():
    user_input = entry_1.get()
    entry_1.delete(0, tk.END)
    
    if user_input == "quit":
        window.quit()
        return
    
    response = generate_response(model, intents, user_input, all_words, tags)
    speak(response, rate=150)
    entry_2.config(state=tk.NORMAL)
    entry_2.insert(tk.END, f"You: {user_input}\n")
    entry_2.insert(tk.END, f"{bot_name}: {response}\n")
    entry_2.config(state=tk.DISABLED)
    entry_2.see(tk.END)
    
def trigger_send_message(event):
    send_message()
# Main GUI setup
window = tk.Tk()
img = PhotoImage(file="C:\\Users\\HP\\Downloads\\physics.png")
window.iconphoto(False, img)
window.geometry("1014x501")
window.configure(bg = "#FFFFFF")
window.title('Proton')

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=501,
    width=1014,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

image_image_1 = PhotoImage(
    file=relative_to_assets("F:/G/build/assets/frame0/image_1.png"))

image_1 = canvas.create_image(
    507.0,
    250.0,
    image=image_image_1
)
entry_image_1 = PhotoImage(
    file=relative_to_assets("F:/G/build/assets/frame0/entry_1.png"))

entry_bg_1 = canvas.create_image(
    430.5,
    431.0,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#121010",
    fg="#ffffff",
    highlightthickness=0,
    font=('verdana', 20, 'bold'),
)
entry_1.place(
    x=48.0,
    y=406.0,
    width=765.0,
    height=48.0
)

entry_1.bind("<Return>", trigger_send_message)
entry_image_2 = PhotoImage(
    file=relative_to_assets("F:/G/build/assets/frame0/entry_2.png"))

entry_bg_2 = canvas.create_image(
    510.0,
    246.0,
    image=entry_image_2
)

entry_2 = Text(
    bd=0,
    bg="#0A0909",
    fg="#ffffff",
    highlightthickness=0,
    font=('verdana', 13, 'bold'),
)
entry_2.place(
    x=46.0,
    y=130.0,
    width=928.0,
    height=230.0
)

button_image_1 = PhotoImage(
    file=relative_to_assets("F:/Proton_Gui/build/assets/frame0/button_1.png"))

send_button = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    relief="flat",
    command=send_message,
)
send_button.place(
    x=906.0,
    y=406.0,
    width=50.17254638671875,
    height=50.10823059082031
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = load_data(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Proton"
speak('hello my name is proton',rate=150)

window.resizable(False, False)
window.mainloop()
