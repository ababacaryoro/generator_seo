import json
from bs4 import BeautifulSoup
from googlesearch import search
import requests
import re
import streamlit as st
from collections import Counter
import numpy as np

container_name = "apps-feedback"
data_path = "app/data/"


def get_semrank_result(keyword: str) -> dict:
    """ Search for Semrank result on specific keywrods and parse it"""
    r = requests.post(url="https://semrank.io/admin/api/keywords", data={"query": keyword})
    return json.loads(r.content)


def get_complement_keywords(keyword):
    """
    Pour trouver les mots clés complémentaires à rajouter dans le contenu
    Args:
        keyword:

    Returns:

    """
    list_complement = []
    try:
        r = requests.post(url="https://semrank.io/admin/api/complement",
                          data={"search": keyword, "spec": "words"})
        l_kw = json.loads(r.content)
        list_complement = sum([i for i in json.loads(l_kw["datas"]["result"]).values()], [])
    except:
        pass

    return list_complement


def get_hn_structure_and_content(url):
    res = {"title": "",
           "description": "",
           "hn_structure": "",
           "content": ""
           }
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a réussi

        soup = BeautifulSoup(response.content, 'html.parser')
        # Extraction du titre de la page
        page_title = soup.title.string.strip() if soup.title else ''

        # Extraction de la description de la page
        meta_description = soup.find('meta', attrs={'name': 'description'})
        page_description = meta_description['content'].strip() if meta_description else ''

        # Extraction de la structure Hn et du contenu associé
        hn_structure = []
        current_section = None

        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = tag.name

            # Trouver le contenu de la section
            content = []
            next_node = tag.find_next_sibling()
            while next_node and (
                    not next_node.name or not next_node.name.startswith('h') or int(next_node.name[1]) > int(level[1])):
                content.append(next_node.get_text(strip=True))
                next_node = next_node.find_next_sibling()

            # Joindre le contenu en une seule chaîne de texte
            content_text = ' '.join(content)
            current_section = (tag.name.upper(), tag.get_text(strip=True), content_text)
            hn_structure.append(current_section)

        all_content = ""
        for level, title, content in hn_structure:
            all_content += f"{level.upper()}: {title} \n\n {content}"

        res = {"title": page_title,
               "description": page_description,
               "hn_structure": hn_structure,
               "content": all_content
               }
    except:
        pass

    return res


def parse_semrank_object(results, top_concurrent=3):
    # 1. related questions
    related_questions = [i["question"] for _, i in results["datas"]["paa"].items()]
    other_related_questions = [i["query"] for _, i in results["datas"]['related'].items()]

    # 2. Concurrent + docapposte data content => Title, snippet, Content, Hn structure
    concurrents_data = []
    pos_concurrents = [c["position"] for _, c in results["datas"]["concurrents"].items()]
    pos_backlink = [c["position"] for _, c in results["datas"]["backlinks"].items()]
    positions = sorted(pos_concurrents + pos_backlink)
    docaposte_data = []
    all_url = []

    # Data on "concurrents" key
    for _, concurrent in results["datas"]["concurrents"].items():
        all_url.append(concurrent["url"])
        a = {"position": concurrent["position"],
             "title": concurrent["title"],
             "snippet": concurrent["snippet"],
             "url": concurrent["url"],
             "descr": concurrent["descr"],
             "headings": concurrent["headings"],
             "nb_words": concurrent["nb_words"],
             "content": concurrent["content"]
             }
        if (concurrent["position"] in positions[:top_concurrent + 1] and
                len(concurrents_data) < top_concurrent):
            concurrents_data.append(a)
        if "docaposte" in concurrent["url"]:
            docaposte_data.append(a)

    # Data on 'backlinks' key
    for _, b in results["datas"]["backlinks"].items():
        # Completer objet si pas suffisant
        if (b["position"] in positions[:top_concurrent + 1] and
                len(concurrents_data) < top_concurrent):
            # Récupérer content et autres si absents
            if ("content" not in b) or ("descr" not in b):
                url_data = get_hn_structure_and_content(b["url"])
                b["descr"] = url_data["description"]
                b["content"] = url_data["content"]
                b["nb_words"] = len(url_data["content"].split(' '))

            a = {"position": b["position"],
                 "title": b["title"],
                 "snippet": b["snippet"],
                 "url": b["url"],
                 "descr": b["descr"] if "descr" in b else "",
                 "headings": b["headings"],
                 "nb_words": b["nb_words"] if "nb_words" in b else 0,
                 "content": b["content"] if "content" in b else ""
                 }
            concurrents_data.append(a)

            if "docaposte" in b["url"]:
                docaposte_data.append(a)
        else:
            continue

    # Ordonner la liste des concurrents
    list_pos_init = [c['position'] for c in concurrents_data]
    list_pos_ord = sorted([c['position'] for c in concurrents_data])
    index_position = [list_pos_init.index(p) for p in list_pos_ord]
    concurrents_data = [concurrents_data[p] for p in index_position]

    obj = {'related_questions': related_questions, 'other_related_questions': other_related_questions,
           'docaposte_data': docaposte_data, 'concurrents_data': concurrents_data, 'all_url': all_url}

    return obj


def google_search(query, site="docaposte.com", num_results=5):
    complete_query = f"{query} site:{site}"
    links = []
    for result in search(complete_query, num_results=num_results, lang="fr"):
        links.append(result)
    return links


def extract_title_and_snippet(link):
    try:
        response = requests.get(link, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title of the page
        title = soup.title.string if soup.title else "No title found"

        # Extract a paragraph or snippet as a summary
        paragraphs = soup.find_all('p')
        snippet = paragraphs[0].get_text() if paragraphs else "No snippet found"

        return title, snippet
    except Exception as e:
        return "No title found", "No snippet found"


def format_ancre(links):
    formatted_links = []
    for link in links:
        title, snippet = extract_title_and_snippet(link)
        # Get last part of the article
        last_part = link.rstrip('/').split('/')[-1]
        ancre = ' '.join(last_part.replace('-', ' ').replace('_', ' ').split())

        formatted_links.append(f"""Titre : '{ancre}' | Lien : {link}""")
    return "\n".join(formatted_links)


def add_hn_label(text):
    """
    Rajouter le niveau Hn après les # du markdown
    Args:
        text:

    Returns:

    """

    def remplacer_titre(match):
        # Obtenir le nombre de # pour déterminer le niveau
        niveau = len(match.group(1))
        # Déterminer le titre
        titre = match.group(2).strip()
        # Créer le nouveau titre avec le niveau Hn ajouté
        return f"{match.group(1)} (H{niveau}) {titre} "

    # Expression régulière pour matcher les titres markdown
    pattern = r"(#{1,6})\s*(.+)"
    # Remplacer chaque titre en ajoutant le niveau
    texte_modifie = re.sub(pattern, remplacer_titre, text)

    return texte_modifie.replace("```markdown", "").replace("```", "")


def calculate_semantic_score(text, keywords) -> int:
    """
    Calculer le score sémantique du texte à partir des mots clés donnés => proportion de mots clés existant dans
    le contenu
    Args:
        text:
        keywords:

    Returns: Score entre 0 et 100

    """

    # Lowercase texte
    text = text.lower()

    # Counter for present keywords
    score = 0

    # Loop over keywords and find if present in the content
    for keyword in keywords:
        occurrences = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
        score += (occurrences > 0) * 1

    # Normalizing the score
    max_possible_score = len(keywords)
    normalized_score = (score / max_possible_score) * 100

    return int(min(100, max(0, normalized_score)))


def reset_generation(step="structure"):
    """
    Réinitialiser les choix et les affichages
    Args:
        step: prends les valeurs structure, content

    Returns: None

    """

    if step == "structure":
        st.session_state["content_ready"] = False

        st.session_state["n_content_choix"] = None

    elif step == "content":
        pass
