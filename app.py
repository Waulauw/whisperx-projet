from flask import Flask, request, jsonify
import whisperx
import os

app = Flask(__name__)
model = whisperx.load_model("base")  # tu peux mettre "large-v2" si besoin

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["file"]
    audio_path = "temp.wav"
    audio_file.save(audio_path)
    result = model.transcribe(audio_path)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render fournit le port dans la variable d'environnement
    app.run(host="0.0.0.0", port=port)
