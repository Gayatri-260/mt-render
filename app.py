from flask import Flask, request, jsonify, render_template_string
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<title>Machine Translation</title>
<h2>Translate Text</h2>
<form method="post" action="/translate">
  Source (e.g., en): <input name="source_lang"><br><br>
  Target (e.g., es): <input name="target_lang"><br><br>
  Text:<br><textarea name="text" rows="4" cols="50"></textarea><br><br>
  <input type="submit" value="Translate">
</form>
{% if translation %}
<h3>Translation:</h3>
<p>{{ translation }}</p>
{% endif %}
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/translate", methods=["POST"])
def translate_form():
    src = request.form["source_lang"]
    tgt = request.form["target_lang"]
    text = request.form["text"]
    translation = translate_text(src, tgt, text)
    return render_template_string(HTML_TEMPLATE, translation=translation)

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    src = data["source_lang"]
    tgt = data["target_lang"]
    text = data["text"]
    translation = translate_text(src, tgt, text)
    return jsonify({"translation": translation})

def translate_text(source_lang, target_lang, text):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
