# source for word embeddings:
- german: 
https://www.deepset.ai/german-word-embeddings =>
https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt

# source for deployment
https://realpython.com/flask-by-example-part-1-project-setup/

# source for word embedding techniques
nlp programming course on udemy by "lazy programmer"

# source for free .css stylesheet
colorlib

# free stock photo
from pexels.com

# venv initialisieren
> python3 -m venv venv

# venv aktivieren
> source venv/bin/activate

# dependencies installieren
> pip install -r requirements.txt

# app lokal starten, so dass jede änderung automatisch reloaded wird
> FLASK_ENV=development flask run

# app lokal im browser sehen
http://localhost:5000/

# adress already in use
>  ps -fA | grep flask

# app nach aws pushen (geht seit ca 01.08. nicht mehr, aus Kostengründen Beanstalk und Codepipeline abgeschaltet)
> git push (gibt cicd connection über codepipeline)

# app nach heroku pushen