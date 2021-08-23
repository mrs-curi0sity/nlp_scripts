
import logging
from flask import Flask, request, render_template

from src.embedding_utilities import euclidean_dist, cosine_dist
from src.embedding import Embedding
import pathlib
import os

# logging.basicConfig(encoding='utf-8', level=logging.INFO) # DEBUG gives thousands of DEBUG lines when reading aws s3 files

BUCKET_NAME = 'ma-2021-07-word-embeddings'
FILE_NAME_EN = 'glove.6B.100d_10_test_file.txt' #'glove.6B.100d.txt' #
FILE_NAME_DE = 'GloVe_vectors_de_10_test_file.txt' # 'GloVe_vectors_de_50000.txt' # #'GloVe_vectors_de.txt' big file: 1.3 Mio words, 2GB

"""
if '/Users/magdalena.aretz' in str(pathlib.Path('.').parent.resolve()):
    LOCAL_INSTANCE = True
else:
    LOCAL_INSTANCE = False
"""

LOCAL_INSTANCE = True # dont  use aws s3 bucket. its way to slow. keep a local copy of your embedding files in /data instead

def load_embeddings(is_local = LOCAL_INSTANCE):
    logging.info(f'start loading embeddings. is_local: {LOCAL_INSTANCE}')
    """
    load pretrained word embeddings.
    if this is a local flask server, use local file version
    otherwise: usw (slower) aws s3 bucket file
    """
    if LOCAL_INSTANCE:
        glove_path_en = [os.path.join('data', FILE_NAME_EN)]
        glove_path_de = [os.path.join('data', FILE_NAME_DE)]
    else:
        glove_path_en = [f's3://{BUCKET_NAME}/{FILE_NAME_EN}']
        glove_path_de = [f's3://{BUCKET_NAME}/{FILE_NAME_DE}']
    glove_embedding_en = Embedding(language='en', path_list=glove_path_en)
    glove_embedding_de = Embedding(language='de', path_list=glove_path_de)
    
    return glove_embedding_en, glove_embedding_de
       
glove_embedding_en, glove_embedding_de = load_embeddings(is_local = LOCAL_INSTANCE)


# in heroku: should be called app, in aws should be called application
application = Flask(__name__)
app = application

@application.route("/")
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

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
