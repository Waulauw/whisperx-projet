from flask import Flask, request, jsonify
import whisperx
import os # Assure-toi que cette ligne est bien là !

app = Flask(__name__)
# Tu peux essayer "large-v2" pour de meilleurs résultats, mais "base" est plus rapide et moins gourmand.
# Pour Render sans GPU, "base" ou "small" sont plus raisonnables.
model = whisperx.load_model("base", device="cpu") # Spécifier device="cpu" est plus sûr sur Render

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # Sauvegarder temporairement le fichier
        # Il est important de gérer la suppression du fichier temporaire après usage
        temp_audio_path = "temp_audio_file" # Pas besoin d'extension si whisperx le gère
        file.save(temp_audio_path)

        try:
            # Transcrire avec WhisperX
            # result = model.transcribe(temp_audio_path)
            # Pour obtenir les mots et leurs timestamps, il faut charger un modèle d'alignement
            audio = whisperx.load_audio(temp_audio_path)
            result = model.transcribe(audio)
            
            # Optionnel mais recommandé pour une meilleure précision des mots:
            align_model, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
            result = whisperx.align(result["segments"], align_model, metadata, audio, device="cpu", return_char_alignments=False)
            # result contiendra maintenant des segments avec des listes de 'words' et leurs 'start'/'end' timestamps

        except Exception as e:
            # Nettoyage en cas d'erreur
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({"error": str(e)}), 500
        
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return jsonify(result) # 'result' devrait contenir les segments et les mots avec timestamps

if __name__ == "__main__":
    # Render fournira le port via la variable d'environnement PORT
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)
