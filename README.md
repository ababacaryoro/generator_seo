## Introduction 
Cette application est une aide à la rédaction de contenu pour un article de blog. À partir d'un mot clé (un sujet), l'application propose une structure HN pour l'article à rédiger, puis le contenu de l'article, le titre et un résumé.

## Fonctionnement global

Cette application permet de générer du contenu pour écrire un article de blog. Le point de départ est un mot clé ou un sujet. A partir de cela, l'application procède par 6 étapes : 

- Etape 1 : Recherche dans le navigateur les 10 pages sur le sujet les mieux référencés pour récupérer  leur structure HN, leur contenu, les mots clés importants et d'autres informations complémentaires comme les autres questions que peuvent se poser les internautes autour du sujet, l'intention de recherche que peut avoir un internaute sur ce sujet et un listing des pages existantes de notre entreprise en lien avec le sujet.

- Etape 2 : Proposition d'une structure HN de départ. L'application se base sur le mot clé fourni par l'utilisateur et de l'intention de recherche que peut avoir un internaute sur ce sujet pour proposer une première version de structure HN très générale.

- Etape 3 : Génération de la structure de l'article. L'application se base sur la structure initiale proposée, les structures des 10 pages des "concurrents" et les autres questions que peuvent se poser les internautes autour du sujet pour proposer 3 options de nouvelle structure originale pour l'article. L'utilisateur en choisi 1 pour la suite.

- Etape 4 : Génération du contenu. L'application utilise en input la structure HN sélectionnée,  les autres questions posées par les internautes autour du sujet, les mots clés importants pour le sujet, l'intention de recherche que peut avoir un internaute sur ce sujet, le  listing des pages existantes de Docaposte en lien avec le sujet, les contenus des 10 pages "concurrentes". A partir de tout cela, 3 versions d'article de blog sont générées de manière séquentielle, partie par partie en respectant la structure HN choisie. Les 3 versions d'article sont ensuite reformulées pour adapter le style et le contenu à des contraintes fixées en amont. L'utilisateur choisi alors 1 versions qui va être utilisée pour la suite

- Etape 5 : Génération du titre et de la description. Sur la base du contenu choisi, l'application propose 3 versions de titre et description. L'utilisateur en choisi 1 pour la suite.

- Etape 6 : Génération du résumé. Sur la base toujours du contenu choisi, l'application propose 3 versions de résumés et l'utilisateur en choisi 1.

A la fin de toutes ces étapes, une récapitulatif des choix de l'utilisateur est proposé avec le titre et la description, la structure HN, le contenu de l'article et le résumé.

NB : il faut noter que dans la dernière version, seule la génération du contenu est dépendante de celle de la structure. La génération de titre ou résumé peut se faire de manière indépendante.

## Spécificités techniques

Techniquement, l'application se base entièrement sur du prompt engineering pour proposer les options de structure, contenu etc. Mais avant cela, le mot clé est recherché sur le navigateur pour récupérer les premiers contenus qui viennent à l'internaute. Ces pages sont scrapées en passant par des APIs de SEMRANK (https://semrank.io/admin/api/keywords pour l'essentiel et https://semrank.io/admin/api/complement pour les mots clés complémentaires).


## Structure du Projet

La structure de votre projet est la suivante :

```
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── app.py
│   ├── assets
│   │   ├── css
│   │   │   ├── theme-old.css
│   │   │   └── theme.css
│   │   └── images
│   │       ├── logo.png
│   │       └── worflow_appli.png
│   ├── data
│   ├── model
│   │   ├── __init__.py
│   │   ├── gpt
│   │   │   ├── __init__.py
│   │   │   └── get_model.py
│   │   ├── mistral
│   │   │   ├── __init__.py
│   │   │   └── get_model.py
│   │   └── prompt
│   │       ├── __init__.py
│   │       ├── prompt_template.py
│   └── utils
│       ├── banned_expression.yml
│       ├── llmtools.py
│       ├── pdf_loader.py
│       ├── textformatools.py
│       ├── tools.py
│       ├── ui.py
│       └── vault.py
├── azure-pipelines.yml
├── config.yml
├── notebooks
│   ├── 00-data.ipynb
│   ├── 01-creation-contenu.ipynb
│   ├── 02-tests.ipynb
│   └── 03-tests-structure-contenu.ipynb
├── poetry.lock
├── pyproject.toml
├── requirements.txt
└── tests
    └── __init__.py

```

## Étapes pour exécuter le projet 

1. **Cloner le repo** :

    ```sh
    git clone git@github.com:ababacaryoro/generator_seo.git
    cd generator_seo
    ```

2. **Gestion des clés d'accès et librairies**

Sachant qu'il est nécessaire d'avoir des clés pour utiliser les LLM, le déploiement doit se faire 
dans un environnement qui contient les variables **MIXTRAL_API_BASE**, **MIXTRAL_API_KEY** et **GPT_API_KEY**. Cela peut se gérer par 
un key vault sur Azure (le dockerfile et la pipeline Azure sont configurés pour) ou bien en exportant ces variables 
dans le dockerfile si c'est en local (charger un fichier .env et exporter). Cette dernière option est moins "propre"
pour une industrialisation

   ```sh
   export MIXTRAL_API_BASE="VALEUR_A_COMPLETER" # => le lien vers le déploiement de MISTRAL
   export MIXTRAL_API_KEY="VALEUR_A_COMPLETER"  # =>  la clé de la souscription à MISTRAL
   export GPT_API_KEY="VALEUR_A_COMPLETER"  # =>  la clé d'accès à l'API GPT 
   ```

Il faudra également installer les librairies nécessaires pour l'exécution de l'application. Pour cela, commencez par utiliser un environnement virtuel avec la version *3.10.14 de Python*. Ensuite, pour les dépendances, cela peut se faire de 2 manières : avec *poetry* ou avec le fichier *requirements.txt*

Avec poetry, un packaging des librairies est déjà disponible dans les fichiers du projet (*poetry.lock* et *pyproject.toml*). Il suffit donc de lancer les commandes ```pip install poetry ``` et ```poetry install  ``` pour que les librairies utiles soient installées.

Avec le fichier *requirements.txt*, il y a un listing des principales librairies complémentaires à rajouter. Leur installation peut se faire avec la commande ```pip install -r /path/to/requirements.txt```


Dans le cas où il est nécessaire de refaire l'environnement virtuel avec les librairies, la démarche suivante pourrait être adoptée : 

- Créer l'environnement virtuel avec la version *3.10.14 de Python* 
- Exécuter les commandes suivantes : 
    ```sh
    peotry init
    poetry add  torch==2.2.2 
    poetry add $(cat requirements.txt)
    ```

3. **Lancement de l'application en local** :

L'application peut être lancée en local directement en se positionnant sur le dossier AgentSEO et en exécutant la commande sh  ```streamlit run app/app.py  ```. 

4. **Construire et exécuter le conteneur docker** :

    ```sh
    docker build -t seo .
    docker run -p 80:8501 seo
    ```

Une fois le conteneur en cours d'exécution, vous pouvez accéder à l'application Streamlit avec l'URL
suivante en local : `http://localhost:80`.


# Contributeurs

- [Ababacar BA](mailto:yoroba93@gmail.com)


# Aide
- [Ababacar BA](mailto:yoroba93@gmail.com)
