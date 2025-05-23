from flask import Flask, request, jsonify
import whisperx
import os
import logging # Mieux que print pour le logging

# Configure logging pour voir les messages dans les logs de Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Flask app script starting...") # Log au tout début

app = Flask(__name__)
logger.info("Flask instance created.")

try:
    logger.info("Attempting to load WhisperX ASR model (base)...")
    # Ajout de compute_type="float32" pour la compatibilité CPU
    model = whisperx.load_model(
        "base", 
        device="cpu", 
        compute_type="float32"  # Correction pour l'erreur float16
    ) 
    logger.info("WhisperX ASR model loaded successfully.")
    
    # Le modèle d'alignement est chargé dynamiquement par langue dans la route
    # Pas besoin de le précharger ici globalement pour toutes les langues.

except Exception as e:
    logger.error(f"CRITICAL ERROR during ASR model loading: {e}", exc_info=True)
    # Si une erreur se produit ici, l'application ne démarrera pas correctement.
    # Gunicorn pourrait ne pas pouvoir démarrer.
    raise # Relance l'exception pour que le processus échoue et que Gunicorn le logue si possible

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    logger.info("Received request to /transcribe endpoint.")
    if 'file' not in request.files:
        logger.warning("No 'file' part in request form-data.")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No file selected (filename is empty).")
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        temp_audio_path = "temp_audio_file_upload" # Nom de fichier temporaire
        logger.info(f"Saving uploaded file temporarily to: {temp_audio_path}")
        file.save(temp_audio_path)
        logger.info("File saved successfully.")

        try:
            logger.info(f"Loading audio from: {temp_audio_path}")
            audio = whisperx.load_audio(temp_audio_path)
            logger.info("Audio loaded. Starting transcription...")
            
            # 1. Transcrire l'audio en segments
            result_segments = model.transcribe(audio)
            logger.info(f"Transcription to segments complete. Language detected: {result_segments.get('language', 'N/A')}")
            
            # 2. Aligner les segments transcrits pour obtenir les timestamps par mot
            detected_language = result_segments.get("language")
            if not detected_language:
                logger.error("Language detection failed after transcription.")
                # Essayer de supprimer le fichier temporaire même en cas d'erreur ici
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return jsonify({"error": "Language detection failed"}), 500

            logger.info(f"Loading alignment model for language: {detected_language}")
            # Pas besoin de compute_type pour load_align_model, device="cpu" suffit.
            align_model, metadata = whisperx.load_align_model(language_code=detected_language, device="cpu")
            logger.info("Alignment model loaded. Starting word alignment...")
            
            aligned_result = whisperx.align(
                result_segments["segments"], 
                align_model, 
                metadata, 
                audio, 
                device="cpu", 
                return_char_alignments=False
            )
            logger.info("Word alignment complete.")
            # `aligned_result` contient les segments avec les listes de 'words' et leurs timestamps.

        except Exception as e:
            logger.error(f"Error during transcription/alignment process: {e}", exc_info=True)
            # Assurer le nettoyage en cas d'erreur
            if os.path.exists(temp_audio_path):
                logger.info(f"Cleaning up temp file due to error: {temp_audio_path}")
                os.remove(temp_audio_path)
            return jsonify({"error": "Error during transcription process", "details": str(e)}), 500
        
        finally:
            # Assurer le nettoyage du fichier temporaire dans tous les cas (succès ou échec géré)
            if os.path.exists(temp_audio_path):
                logger.info(f"Cleaning up temp file: {temp_audio_path}")
                os.remove(temp_audio_path)
            
        logger.info("Successfully processed request. Returning aligned result.")
        return jsonify(aligned_result) 
    
    # Ce cas ne devrait pas être atteint si la logique de fichier est correcte
    logger.error("File processing logic error, request not handled.")
    return jsonify({"error": "Server-side file processing error"}), 500

if __name__ == "__main__":
    # Cette section est pour le développement local, Gunicorn gère cela en production sur Render.
    logger.info("Running Flask app directly (for local development).")
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True est utile localement

logger.info("Flask app script definition complete. If run by Gunicorn, Gunicorn takes over now.")
