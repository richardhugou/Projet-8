#!/bin/bash

# Script de test de fumée (Smoke Test) pour valider l'API dans Docker
# Ce script vérifie que l'image se construit, démarre et répond correctement
# avec les données du domaine (Scoring).

# 1. Configuration
IMAGE_NAME="credit-api:test"
CONTAINER_NAME="credit-api-smoke-test"
PORT=8001 # Utiliser un port différent pour éviter les conflits

echo "DÉBUT DU TEST DE FUMÉE DOCKER"

# 2. Construction de l'image
echo "Construction de l'image $IMAGE_NAME..."
export DOCKER_BUILDKIT=0
docker build -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "ÉCHEC de la construction de l'image."
    exit 1
fi

# 3. Démarrage du conteneur
echo "🏃 Démarrage du conteneur en arrière-plan..."
# On s'assure qu'aucun ancien conteneur ne traîne
docker rm -f $CONTAINER_NAME 2>/dev/null
docker run -d --name $CONTAINER_NAME -p $PORT:8000 $IMAGE_NAME

# Attendre que l'API soit prête (Healthcheck simplifié)
echo "Attente du démarrage de l'API (max 20s)..."
MAX_RETRIES=20
COUNT=0
until $(curl -sSf http://localhost:$PORT/ > /dev/null); do
    printf '.'
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -eq $MAX_RETRIES ]; then
        echo "ÉCHEC : L'API n'a pas répondu à temps."
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME
        exit 1
    fi
done
echo "API Up !"

# 4. Test de Prédiction (Domaine Métier)
echo "Envoi d'une requête de prédiction..."
RESPONSE=$(curl -s -X POST "http://localhost:$PORT/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "AMT_INCOME_TOTAL": 150000,
           "AMT_CREDIT": 450000,
           "AMT_ANNUITY": 20000,
           "DAYS_BIRTH": -18000
         }')

echo "Réponse reçue : $RESPONSE"

# Vérification du domaine (Le seuil est à 0.091)
if [[ $RESPONSE == *"probability_default"* ]] && [[ $RESPONSE == *"status"* ]]; then
    echo "SUCCÈS : Le domaine métier répond correctement dans Docker."
else
    echo "ÉCHEC : La réponse est invalide ou incomplète."
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
    exit 1
fi

# 5. Nettoyage
echo "Nettoyage du conteneur..."
docker rm -f $CONTAINER_NAME
echo "TEST DE FUMÉE TERMINÉ AVEC SUCCÈS"
