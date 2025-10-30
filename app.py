from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    try:
        source_lang = request.form["source_lang"]
        target_lang = request.form["target_lang"]
        text = request.form["text"]

        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        return render_template("index.html", result=output)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")
