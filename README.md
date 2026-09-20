# Prompt Engineering Trainer — Atelier MVA

Application Streamlit permettant à un apprenant de construire son propre
prompt système (Canvas en 8 composants), de le tester sur un domaine
d'étude au choix via l'API Groq (Llama 3.3), de visualiser le **prompt
exact envoyé** à chaque question, et de suivre la **consommation de
tokens** en temps réel.

## Fichiers

- `prompt_engineering_trainer.py` — application principale.
- `requirements.txt` — dépendances (streamlit, groq, tiktoken).

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run prompt_engineering_trainer.py
```

Saisissez votre clé API Groq (obtenue sur https://console.groq.com) dans
la barre latérale.

## Déployer sur Streamlit Community Cloud

1. Poussez ce dossier dans un dépôt GitHub :

   ```bash
   git init
   git add .
   git commit -m "Prompt Engineering Trainer"
   git branch -M main
   git remote add origin <URL_DE_VOTRE_DEPOT>
   git push -u origin main
   ```

2. Sur [streamlit.io/cloud](https://streamlit.io/cloud), créez une nouvelle
   application en pointant vers ce dépôt et le fichier
   `prompt_engineering_trainer.py`.

3. (Recommandé pour un atelier) Dans **Settings → Secrets** de
   l'application déployée, ajoutez :

   ```toml
   GROQ_API_KEY = "gsk_votre_cle"
   ```

   L'application détecte automatiquement `st.secrets["GROQ_API_KEY"]` et
   masque le champ de saisie manuelle — pratique si vous fournissez une
   clé partagée aux participants d'un atelier plutôt que de leur demander
   de créer un compte Groq individuellement.

## Notes pédagogiques

- Le comptage de tokens utilise l'encodeur `cl100k_base` (tiktoken) comme
  **approximation** — le tokenizer réel de Llama 3.3 diffère légèrement.
  Les tokens affichés après chaque réponse (« Tokens — entrée / sortie /
  total ») proviennent en revanche directement du champ `usage` renvoyé
  par l'API Groq et sont donc exacts.
- L'expander « Voir le prompt exact envoyé » sous chaque réponse est le
  support direct des Ateliers 4 et 5 (LO4/LO5) : il permet de comprendre
  pourquoi un modèle a répondu de telle façon, et de repérer les cas où
  le prompt système n'empêche pas une dérive.
