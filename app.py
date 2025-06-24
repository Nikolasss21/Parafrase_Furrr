from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoTokenizer
import threading
import os

app = Flask(__name__)

# Inisialisasi model dan tokenizer
model_loaded = False
tokenizer = None
paraphraser = None
processing_lock = threading.Lock()

def load_model():
    """Memuat model parafrase"""
    global model_loaded, tokenizer, paraphraser
    if not model_loaded:
        print("Memuat model parafrase...")
        model_name = "cahya/t5-base-indonesian-paraphrase"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        paraphraser = pipeline('text2text-generation', model=model_name)
        model_loaded = True
        print("Model berhasil dimuat!")

def chunk_text(text, max_chunk_size=300):
    """Membagi teks besar menjadi bagian-bagian kecil"""
    if not tokenizer:
        load_model()
    
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        
        if current_length >= max_chunk_size:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    
    return chunks

def paraphrase_large_text(text):
    """Memparafrase teks besar dengan chunking"""
    chunks = chunk_text(text)
    paraphrased_chunks = []
    
    for chunk in chunks:
        with processing_lock:
            result = paraphraser(
                f"parafrase: {chunk}",
                max_length=512,
                num_beams=5,
                num_return_sequences=1
            )[0]['generated_text']
            paraphrased_chunks.append(result.replace("parafrase: ", ""))
    
    return " ".join(paraphrased_chunks)

@app.route('/', methods=['GET'])
def index():
    """Menampilkan halaman utama"""
    return render_template('index.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Endpoint untuk memproses parafrase"""
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400
    
    # Hitung jumlah kata
    word_count = len(text.split())
    if word_count > 2000:
        return jsonify({'error': 'Teks melebihi batas 2000 kata'}), 400
    
    try:
        load_model()  # Pastikan model dimuat
        result = paraphrase_large_text(text)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(threaded=True, port=5000, debug=True)
