import json
import requests
import os
from bs4 import BeautifulSoup
import ast
import re


def parse_hn_from_dict(obj):
    res = [(h["type"], h["text"]) for _, h in obj.items()]

    return res


def hn_list_to_dict(obj) -> dict:
    res = dict([(i, {obj[i][0], obj[i][1]}) for i in range(len(obj))])

    return res


def raw_md_to_list_hn(markdown_text):
    """
    Extraire structure HN initialement en markdown pour la transformer en liste de liste [niveau, texte]

    Args:
        markdown_text: texte en input

    Returns: Structure HN en liste

    """
    # Utiliser une expression régulière pour identifier les titres Hn
    hn_pattern = re.compile(r"^(#+)\s*(.+)")

    # Initialiser une liste pour stocker la structure Hn
    structure = []

    # Parcourir chaque ligne du texte markdown
    for line in markdown_text.splitlines():
        match = hn_pattern.match(line)
        if match:
            # Obtenir le niveau de l'en-tête (nombre de '#')
            level = len(match.group(1))
            # Obtenir le titre de l'en-tête
            title = match.group(2).strip()
            # Ajouter à la structure en utilisant le format ["hN", "titre"]
            structure.append([f"h{level}", title])

    return structure


def hn_list_to_md(obj, content_exist=False) -> str:
    res = "\n\n".join(["<" + h[0] + ">" + h[1] + "</" + h[0] + ">" for h in obj])
    if content_exist:
        res = "\n\n".join(
            [
                "<" + h[0] + ">" + h[1] + "</" + h[0] + ">" + "<br>" + h[2] + "<br>"
                for h in obj
            ]
        )

    return res


def extract_tuples_from_text(text) -> list:
    s = text.find("[")
    e = text.rfind("]")
    try:
        return ast.literal_eval(text[s : e + 1])
    except:
        return []


def parse_text_to_dict(text: str) -> dict:
    s = text.find("{")
    e = text.rfind("}")
    result_dict = {}
    # try:
    # Nettoyer le texte et transformer les fausses échappements
    clean_text = text[s : e + 1].replace(
        "\\'", "'"
    )  # .replace("('", '("""').replace("',", '""",').replace(", '", ', """') \
    # .replace("')", '""")')
    # Utiliser ast.literal_eval pour évaluer le texte en un dictionnaire Python
    result_dict = ast.literal_eval(clean_text)
    # except:
    #    pass
    return result_dict


def generate_markdown_toc(hn_list):
    toc_lines = []
    # Dictionnaire pour suivre le compteur de chaque niveau de titre
    counters = {f"h{i}": 0 for i in range(1, 7)}
    previous_level = 0

    for level, title in hn_list:
        current_level = int(level[1])

        # Reset des compteurs des sous-niveaux si on change de niveau
        if current_level <= previous_level:
            for i in range(current_level + 1, 7):
                counters[f"h{i}"] = 0

        # Incrément du compteur du niveau actuel
        counters[level] += 1
        previous_level = current_level

        # Génération du numéro basé sur les compteurs
        num = ".".join(str(counters[f"h{i}"]) for i in range(1, current_level + 1))

        # Ajout de l'entrée formatée dans la table des matières
        indent = "    " * (current_level - 1)
        toc_lines.append(f"{indent}{num}. {title}")

    return "\n\n".join(toc_lines)


def html_to_markdown_with_levels(html_text):
    # Utiliser des expressions régulières pour trouver les balises Hn et leur contenu
    pattern = re.compile(r"<(h[1-6])>(.*?)<\/\1>", re.IGNORECASE)
    matches = pattern.findall(html_text)

    # Construire le texte Markdown
    markdown_lines = []
    for match in matches:
        tag, content = match
        level = int(tag[1])
        indent = " " * 4 * (level - 1)  # 4 espaces pour chaque niveau de tabulation
        markdown_lines.append(f"{indent}{tag.upper()}: {content.strip()}")

    # Joindre les lignes avec des sauts de ligne pour créer la structure Markdown finale
    markdown_text = "\n\n".join(markdown_lines)
    return markdown_text


def get_markdown_from_list_tuples(l_tuples):
    # Construire le texte Markdown
    markdown_lines = []
    for tag, title, content in l_tuples:
        level = int(tag[1])
        indent = " " * 4 * (level - 1)  # 4 espaces pour chaque niveau de tabulation
        markdown_lines.append(
            f"{indent}{tag.upper()}: {title.strip()} \n\n {content.strip()}"
        )

    # Joindre les lignes avec des sauts de ligne pour créer la structure Markdown finale
    markdown_text = "\n\n".join(markdown_lines)
    return markdown_text


def parse_structure(text):
    # Diviser le texte en lignes
    lines = text.strip().split("\n")
    result = []

    # Définir une expression régulière pour capturer les niveaux de titre et les titres (insensible à la casse)
    pattern = re.compile(r"^(h[1-6]):\s*(.*)$", re.IGNORECASE)

    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            match = pattern.match(stripped_line)
            if match:
                level = match.group(1).lower()  # Convertir en minuscules (h1, h2, etc.)
                title = match.group(2).strip()  # Extraire le titre
                result.append([level, title])

    return result
