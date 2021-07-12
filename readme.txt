# venv initialisieren
> python3 -m venv venv

# venv aktivieren
> source venv/bin/activate

# dependencies installieren
> pip install -r requirements.txt

# app starten, so dass jede änderung automatisch reloaded 
> FLASK_ENV=development flask run