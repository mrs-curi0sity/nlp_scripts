import logging
from flask import Flask, request, render_template

from embedding_utilities import euclidean_dist, cosine_dist
from embedding import Embedding

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


glove_path_en = '../large-files/glove6B/glove.6B.50d.txt'
glove_path_de = '../large-files/GloVe_vectors_de.txt'

glove_embedding_en = Embedding(language='en', path=glove_path_en)
glove_embedding_de = Embedding(language='de', path=glove_path_de)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("input-king-queen.html")

@app.route("/", methods=['POST'])
def when_posted():
    language = request.form['language']
    begriff_1 = request.form['begriff_1']
    processed_text_1 = begriff_1.lower()
    begriff_2 = request.form['begriff_2']
    processed_text_2 = begriff_2.lower()
    begriff_3 = request.form['begriff_3']
    processed_text_3 = begriff_3.lower()

    logging.info(f'----- received this input: {begriff_1} {begriff_2} {begriff_3}')
    if language.strip() =='en':
        analogy = glove_embedding_en.find_analogies(processed_text_1, processed_text_2, processed_text_3)
    elif language.strip() =='de':
        analogy = glove_embedding_de.find_analogies(processed_text_1, processed_text_2, processed_text_3)
    else:
        return "sorry. language unknown"
    logging.info(f'----- computed this analogy: {analogy}')
    return render_template("input-king-queen.html", begriff_1=begriff_1, begriff_2=begriff_2, begriff_3=begriff_3, begriff_4=analogy)

app.run(host='localhost', port=5000)