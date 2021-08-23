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



# app nach ec2 pushen
https://www.twilio.com/blog/deploy-flask-python-app-aws
1. per ssh zu EC2 instanz verbinden mit
> ssh ubuntu@<YOUR_IP_ADDRESS>    also 
> ssh ubuntu@3.67.204.47 (IP aus AWS console => ec2 instanzen abgeguckt)

2. dateien syncen entweder via git
> git clone / pull / push
oder via rsync
> sudo rsync -rv <FULL_PATH>/ ubuntu@<YOUR_IP_ADDRESS>:/home/ubuntu/deployedapp
> sudo rsync -rv /Users/magdalena.aretz/code/nlp_scripts/data/glove.6B.50d.txt ubuntu@3.67.204.47:/home/ubuntu/nlp_scripts/data/glove.6B.50d.txt

3. tmux session aufmachen
> tmux new -s test-deployment

4. app starten
python3 app.py

5. status checken
http://ec2-3-69-27-90.eu-central-1.compute.amazonaws.com:5000/


16. detach tmux
> ctrl + D

17. reaattach tmux
> tmux attach -t test-deployment