# Utilise une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Définir la commande par défaut pour exécuter le script d'entraînement
CMD ["python", "src/train.py"]
