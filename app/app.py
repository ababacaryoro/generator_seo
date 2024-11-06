import os
import time
import numpy as np
import streamlit as st
import yaml
from yaml.loader import SafeLoader

from model.gpt.get_model import CustomGPT
from model.mistral.get_model import instantiate_mixtral
from utils.llmtools import (
    StructureHn,
    get_intention_recherche,
    build_chain,
    generate_prompt_title,
    get_init_structure,
    generate_prompt_reformulate_content,
    PartContent,
    Summary,
    get_summary,
    Title,
    get_all_content_proposals,
    generate_prompt_summary,
    generate_structure_proposals,
)
from utils.textformatools import (
    parse_hn_from_dict,
    parse_structure,
    hn_list_to_md,
    html_to_markdown_with_levels,
    raw_md_to_list_hn,
)
from utils.tools import (
    parse_semrank_object,
    get_semrank_result,
    google_search,
    format_ancre,
    add_hn_label,
    get_complement_keywords,
    calculate_semantic_score,
    reset_generation,
)

# --
with open("config.yml") as f:
    config = yaml.load(f, Loader=SafeLoader)
os.environ["OPENAI_API_URL"] = config["OPENAI_API_URL"]

TOP_CONCURRENTS = config["TOP_CONCURRENTS"]
TOP_CONTENT_PROMPT = 5  ## TODO => Adapter en fonction de la taille des contenus =>
# Prévoir :
#   - 3 contenus à générer + taille prompt == environ 5 * 1500 tokens => 8k tokens environs
#   - X contenus des concurrents (trouver x) == X * nb_token_moyen_contenu
#   - max_token === 128k ou 32k suivant le modèle
#   => 8K + X * nb_token_moyen_contenu = 32k => X = (24k )/nb_token_moyen_contenu
ADJUST_PARAM_TOP_CONTENT = 24000.0
MAX_LEN_SUMMARY = 200
version = "0.1"
font_col_txt = "#6396e2ff"

# => LLM
gpt_llm = CustomGPT()
mistral_llm = instantiate_mixtral()


def main():
    # Prendre toute la page
    st.set_page_config(layout="wide")
    # Margin of main
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session vars
    none_args = [
        "keyword",
        "page_cible",
        "contrat_doc_txt",
        "res_struct_llm",
        "page_title",
        "page_description",
        "llm",
        "gpt_llm",
        "mistral_llm",
        "result",
        "intention_recherche",
        "n_structure_choix",
        "n_content_choix",
        "n_title_choix",
        "n_summary_choix",
        "maillage_interne",
        "init_structure_raw_md",
    ]
    for v in none_args:
        if v not in st.session_state:
            st.session_state[v] = None

    list_args = [
        "messages",
        "res_struct_llm",
        "concurrents_hn_structure",
        "related_questions",
        "hn_structure",
        "structure_suggests",
        "selected_structure",
        "content_suggests",
        "concurrents_content",
        "selected_structure",
        "keywords_list",
        "concurrents_url",
        "concurrents_title",
        "concurrents_desc",
        "selected_content",
        "title_suggests",
        "selected_title",
        "summary_suggests",
        "selected_summary",
        "concurrents_data",
    ]
    for arg in list_args:
        if arg not in st.session_state:
            st.session_state[arg] = []
    bool_args = ["structure_ready", "content_ready", "title_ready", "summary_ready"]
    for arg in bool_args:
        if arg not in st.session_state:
            st.session_state[arg] = False

    # Set CSS
    with open("app/assets/css/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Set sidebar
    with st.sidebar:
        st.sidebar.markdown(
            """
            <style>
            .sidebar-title {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 10px; /* Ajustez la hauteur en fonction de vos besoins */
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """
        <div style="text-align: center;">
            <h2 class="sidebar-title">SEO-Accélerateur</h2>
            <br/><br/>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """
        <style>
        .sidebar-title {
            font-size: 45px !important;
            margin-bottom: 0px; /* Réduire la marge inférieure */
            margin-top: 0; /* Supprimer la marge supérieure */
        }
        .subtitle {
            margin-top: 0; /* Supprimer la marge supérieure */
            margin-bottom: 5px; /* Réduire la marge inférieure */
        }
        
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: stretch;
        }
        
        input[type="radio"]{
            accent-color:green;
            background-color: #0000FF !important;
            color: #000000 !important;
        }
        
        div.row-widget.stRadio {
            background-color: #0000FF !important;
            color: #FFFFFF !important;
        }
        
        /* Pour radio-button */
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]  {
             background-color: #9AC5F4;
             padding-right: 10px;
             padding-left: 4px;
             padding-bottom: 3px;
             margin: 4px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        # Model
        model_choice = st.selectbox(
            "Modèle à utiliser 👉",
            ("gpt", "autre (non dispo actuellement)"),
            key="model_type",
            index=0,
            placeholder="Choix du modèle...",
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        with st.columns(3)[1]:
            st.image("app/assets/images/logo.png", width=100)
            # st.markdown("<br><br>", unsafe_allow_html=True)
            st.image("app/assets/images/logo.png", width=100)
            st.markdown("<br><br>", unsafe_allow_html=True)

    # Upload documents
    st.markdown(
        "<h1 style='color: #265495;'>⚙️ Configurer la recherche </h1>",
        unsafe_allow_html=True,
    )

    # Pour les boutons
    st.markdown(
        """
        <style>
        .element-container:has(style){
            display: none;
        }
        #button-after {
            display: none;
        }
        .element-container:has(#button-after) {
            display: none;
        }
        .element-container:has(#button-after) + div button {
            display: inline-block;
            width:250px;
            height:100px;
            padding: 10px 20px;
            font-size: 15px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    keyword = st.text_input(
        "Mot clé à rechercher", "Comment faire une signature électronique?"
    )
    st.session_state.keyword = keyword
    c1, _, c2 = st.columns(3)

    st.session_state["mistral_llm"] = mistral_llm
    st.session_state["gpt_llm"] = gpt_llm
    st.session_state["llm"] = mistral_llm if model_choice == "mistral" else gpt_llm

    # Fonction pour récupérer les données
    def get_data():
        with st.spinner("Recherche des données d'input ..."):
            # Traitement préliminaires
            # choix du model
            start = time.time()
            # Recherche mot clé sur semrank
            results = get_semrank_result(st.session_state.keyword)
            l_complement_keyword = get_complement_keywords(st.session_state.keyword)
            parsed_res = parse_semrank_object(results, top_concurrent=TOP_CONCURRENTS)
            st.session_state["concurrents_data"] = parsed_res["concurrents_data"]
            # Traitement du résultat

            # => Structures Hn
            concurrents = st.session_state["concurrents_data"]
            n_concurrents = len(concurrents)
            concurrents_hn_structure = [
                parse_hn_from_dict(concurrents[i]["headings"])
                for i in range(n_concurrents)
            ]

            st.session_state.concurrents_hn_structure = concurrents_hn_structure

            # => Related questions
            related_questions = parsed_res[
                "related_questions"
            ]  # + parsed_res["other_related_questions"]
            st.session_state.related_questions = "\n\n".join(related_questions)

            # => Mots clés
            keywords_list = [
                s.split(":")[0]
                for s in list(results["datas"]["keywords_list"].values())
            ]
            keywords_list += l_complement_keyword
            st.session_state["keywords_list"] = ", ".join(keywords_list)

            # => Intention de recherche
            intention_recherche = get_intention_recherche(
                st.session_state["keyword"], st.session_state["llm"]
            )
            st.session_state["intention_recherche"] = intention_recherche

            # => Structure initiale
            init_structure_raw_md = get_init_structure(
                st.session_state["keyword"],
                intention_recherche,
                st.session_state["llm"],
            )

            st.session_state["init_structure_raw_md"] = init_structure_raw_md

            # => Maillage interne
            results_docaposte_content = google_search(st.session_state.keyword)
            maillage_interne = format_ancre(results_docaposte_content)
            st.session_state["maillage_interne"] = maillage_interne.replace(
                "\n", "\n\n"
            )

            end = time.time()
            response_time_sec = end - start
            st.success(f"Données récupérées en {int(response_time_sec)} secondes")

            return parsed_res

    # *********************************************************
    # ===> Proposition de structures
    # *********************************************************
    st.divider()
    generate_struct_btn = st.button("Générer la structure")

    # Génération structure
    if generate_struct_btn:

        if keyword:
            st.session_state.keyword = keyword
            # Test if data available
            if len(st.session_state["concurrents_data"]) < 1:
                parsed_res = get_data()

            with st.spinner(
                "Génération de la structure Hn. Le traitement peut prendre quelques minutes ..."
            ):
                start = time.time()

                # => Structures Hn
                concurrents = st.session_state["concurrents_data"]
                n_concurrents = len(concurrents)
                concurrents_hn_structure = [
                    parse_hn_from_dict(concurrents[i]["headings"])
                    for i in range(n_concurrents)
                ]
                # Chaine
                chain_structure = build_chain(st.session_state["llm"], StructureHn)
                exs = "\n".join(
                    [
                        "## Début exemple \n" + str(i) + "\n## Fin exemple \n "
                        for i in concurrents_hn_structure
                    ]
                )
                exs = exs.replace("(", "[").replace(")", "]")

                try:
                    st.session_state["structure_suggests"] = (
                        generate_structure_proposals(
                            chain_structure,
                            st.session_state.keyword,
                            exs,
                            st.session_state["related_questions"],
                            st.session_state["init_structure_raw_md"],
                            n_proposals=3,
                        )
                    )

                    if len(st.session_state["structure_suggests"]) < 3:
                        st.error(
                            "Erreur dans la génération de structure, veuillez réessayer"
                        )
                        st.session_state["structure_ready"] = False
                    else:
                        st.session_state["structure_ready"] = True
                except:
                    st.error(
                        "Erreur dans la génération de structure, veuillez réessayer"
                    )
                    st.session_state["structure_ready"] = False

                # Réinitialiser valeurs choisies et affichées
                reset_generation(step="structure")

                end = time.time()
                response_time_sec = end - start
                st.success(
                    f"Informations traitées en {int(response_time_sec)} secondes"
                )

        else:
            st.error("Veuillez donner un mot clé")

    # Affichage structure
    if st.session_state["structure_ready"]:

        structure_suggests = st.session_state.structure_suggests

        list_structure_suggest_md = [
            hn_list_to_md(i[1]) for i in structure_suggests.items()
        ]

        txt = f"""<p style="color:{font_col_txt}; font-size: 30px;">Structures Hn suggérées </p>"""
        st.markdown(txt, unsafe_allow_html=True)

        c_s0, c_s1, c_s2, c_s3 = st.columns(4)

        with c_s0:
            with st.expander("Proposition initiale", expanded=True):
                # st.markdown(html_to_markdown_with_levels(list_structure_suggest_md[0]), unsafe_allow_html=True)

                init_structure = raw_md_to_list_hn(
                    st.session_state["init_structure_raw_md"]
                )
                init_structure_md = hn_list_to_md(init_structure)
                prop_structure_0 = st.text_area(
                    "Structure Hn",
                    html_to_markdown_with_levels(init_structure_md),
                    height=450,
                )

        with c_s1:
            with st.expander("Proposition améliorée 1", expanded=True):
                prop_structure_1 = st.text_area(
                    "Structure Hn",
                    html_to_markdown_with_levels(list_structure_suggest_md[0]),
                    height=450,
                )
                prop_structure_1_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_structure_1_btn",
                    use_container_width=True,
                )
                if prop_structure_1_btn:
                    key = list(structure_suggests.keys())[0]
                    selected_structure = structure_suggests[key]
                    try:
                        selected_structure = parse_structure(prop_structure_1)
                    except:
                        st.error(
                            "La structure modifiée n'est pas au bon format, la structure initiale est prise par "
                            "défaut"
                        )
                        pass
                    st.session_state["selected_structure"] = selected_structure
                    st.session_state["n_structure_choix"] = "1"

                    reset_generation(step="structure")

        with c_s2:
            with st.expander("Proposition améliorée 2", expanded=True):
                prop_structure_2 = st.text_area(
                    "Structure Hn",
                    html_to_markdown_with_levels(list_structure_suggest_md[1]),
                    height=450,
                )

                prop_structure_2_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_structure_2_btn",
                    use_container_width=True,
                )
                if prop_structure_2_btn:
                    key = list(structure_suggests.keys())[1]
                    selected_structure = structure_suggests[key]
                    try:
                        selected_structure = parse_structure(prop_structure_2)
                    except:
                        st.error(
                            "La structure modifiée n'est pas au bon format, la structure initiale est prise par "
                            "défaut"
                        )
                        pass
                    st.session_state["selected_structure"] = selected_structure
                    st.session_state["n_structure_choix"] = "2"

                    reset_generation(step="structure")

        with c_s3:
            with st.expander("Proposition améliorée 3", expanded=True):
                prop_structure_3 = st.text_area(
                    "Structure Hn",
                    html_to_markdown_with_levels(list_structure_suggest_md[2]),
                    height=450,
                )

                prop_structure_3_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_structure_3_btn",
                    use_container_width=True,
                )
                if prop_structure_3_btn:
                    key = list(structure_suggests.keys())[2]
                    selected_structure = structure_suggests[key]
                    try:
                        selected_structure = parse_structure(prop_structure_3)
                    except:
                        st.error(
                            "La structure modifiée n'est pas au bon format, la structure initiale est prise par "
                            "défaut"
                        )
                        pass
                    st.session_state["selected_structure"] = selected_structure
                    st.session_state["n_structure_choix"] = "3"

                    reset_generation(step="structure")

        # ****===> Inputs used
        txt = f"""<p style="color:{font_col_txt}; font-size: 20px;"> Inputs utilisés pour générer la structure </p>"""
        st.markdown(txt, unsafe_allow_html=True)
        with st.expander("Afficher les inputs"):
            tab1_struct, tab2_struct, tab3_struct = st.tabs(
                [
                    "Structures Hn utilisées",
                    "Autres questions posées",
                    "Intention de recherche",
                ]
            )
            # Structures
            with tab1_struct:
                list_struct = st.session_state.concurrents_hn_structure
                concurrents = st.session_state["concurrents_data"]
                n_concurrents = len(concurrents)
                concurrents_url = [concurrents[i]["url"] for i in range(n_concurrents)]
                concurrents_position = [
                    concurrents[i]["position"] for i in range(n_concurrents)
                ]
                concurrents_nb_words = [
                    concurrents[i]["nb_words"] for i in range(n_concurrents)
                ]
                list_struct_html = [
                    hn_list_to_md(i) for i in list_struct[:TOP_CONCURRENTS]
                ]
                list_struct_md = [
                    html_to_markdown_with_levels(i) for i in list_struct_html
                ]
                n_struct = len(list_struct_md)

                txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Nombre de structures utilisées : {str(n_struct)} </p>"""
                st.markdown(txt, unsafe_allow_html=True)
                txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Exemples de structures :  </p>"""
                st.markdown(txt, unsafe_allow_html=True)

                choices_struct = sorted(concurrents_position)

                def show_structure(position):
                    index_sel = concurrents_position.index(position)
                    st.markdown("**Url** : " + concurrents_url[index_sel])
                    st.markdown(
                        "**Nombre de mots** : " + str(concurrents_nb_words[index_sel])
                    )

                    st.text_area(
                        "Structure n°" + str(position),
                        list_struct_md[index_sel],
                        height=450,
                        disabled=True,
                    )

                c1, c2, c3 = st.columns(3)
                with c1:
                    show_structure(choices_struct[0])
                with c2:
                    show_structure(choices_struct[1])
                with c3:
                    show_structure(choices_struct[2])

            # Questions frequentes
            with tab2_struct:
                st.markdown(st.session_state.related_questions)

            # Intention de recherche
            with tab3_struct:
                st.markdown(st.session_state["intention_recherche"])

    # *********************************************************
    # ===> Proposition de contenu
    # *********************************************************

    # Génération contenu
    if len(st.session_state["selected_structure"]) > 0:
        st.divider()
        txt = f"""<p style="color:{font_col_txt}; font-size: 30px;">Génération de contenus </p>"""
        st.markdown(txt, unsafe_allow_html=True)
        txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Proposition de structure 
        choisie: {str(st.session_state["n_structure_choix"])} </p>"""
        st.markdown(txt, unsafe_allow_html=True)

        number_input_css = """
        <style>
            .stNumberInput {
                width: 200px;
            }
        </style>
        """

        # Appliquer le CSS
        st.markdown(number_input_css, unsafe_allow_html=True)
        concurrents = st.session_state["concurrents_data"]

        concurrents_content = [i["content"] for i in concurrents]
        concurrents_nb_words = [i["nb_words"] for i in concurrents]
        len_content = int(
            np.mean([n for n in concurrents_nb_words if n > 100])
        )  # Eliminer contenu trop petits
        top_content_prompt_adjusted = min(
            int(ADJUST_PARAM_TOP_CONTENT / len_content), len(concurrents)
        )
        concurrents_content_limited = concurrents_content[:top_content_prompt_adjusted]

        # Mots clés
        mots_cles = st.text_area(
            "Mots clés à utiliser (à séparer par des virgules) :",
            st.session_state["keywords_list"],
            height=100,
        )

        # Question fréquentes
        related_questions = st.text_area(
            "Autres questions fréquentes des internautes :",
            st.session_state["related_questions"],
            height=100,
        )
        # Maillage interne
        maillage_interne = st.text_area(
            "Mots clés pour maillage interne :",
            st.session_state["maillage_interne"],
            height=100,
        )

        # Intention de recherche
        intention_recherche = st.text_area(
            "Intention de recherche sur ce mot clé :",
            st.session_state["intention_recherche"],
            height=100,
        )

        # ****===> Inputs used
        txt = f"""<p style="color:{font_col_txt}; font-size: 20px;"> Récap des inputs à utiliser pour générer le 
        contenu </p>"""
        st.markdown(txt, unsafe_allow_html=True)
        with st.expander("Afficher les inputs"):
            tab1_content, tab2_content, tab3_content, tab4_content = st.tabs(
                [
                    "Mots clés",
                    "Autres questions posées",
                    "Maillage interne",
                    "Intention de recherche",
                ]
            )

            # Mots clés
            with tab1_content:
                st.markdown(mots_cles)

            # Questions frequentes
            with tab2_content:
                st.markdown(related_questions)

            # Maillage interne
            with tab3_content:
                st.markdown(maillage_interne)

            # Intention de recherche
            with tab4_content:
                st.markdown(intention_recherche)

        # ****===> Génération
        if st.button("Générer un contenu", key="generate_content"):
            with st.spinner(
                "Génération des propositions de contenu. Le traitement peut prendre quelques minutes ..."
            ):
                start = time.time()
                exs = "\n".join(
                    [
                        "## Début exemple \n" + str(i) + "\n## Fin exemple \n "
                        for i in concurrents_content_limited
                    ]
                )
                st.session_state["keywords_list"] = mots_cles
                st.session_state["related_questions"] = related_questions
                st.session_state["maillage_interne"] = maillage_interne
                st.session_state["intention_recherche"] = intention_recherche

                chain_part_content = build_chain(st.session_state["llm"], PartContent)

                # Loop over sections to generate content
                # => First draft of content
                try:
                    draft_content_proposals = get_all_content_proposals(
                        chain_part_content,
                        st.session_state.keyword,
                        st.session_state["selected_structure"],
                        exs,
                        st.session_state["intention_recherche"],
                        st.session_state["keywords_list"],
                        st.session_state["related_questions"],
                    )
                    # => Prompt for rephrasing
                    # Supplement prompt pour le style
                    supplement_prompt = [
                        "15. " + s
                        for s in [
                            "L'article final doit ABSOLUMENT contenir plusieurs exemples. Certains exemples doivent être "
                            "basés sur des personas. Par exemple, lorsque l’on parle des documents à signer, on pourrait "
                            "illustrer cela avec le cas d’un RH qui utilise la signature électronique pour simplifier la "
                            "gestion des contrats de travail. Les exemples doivent être présentées de manière subtile, en "
                            "évitant d'utiliser tout le temps l'expression 'Par exemple ... '.",
                            "L'article final doit ABSOLUMENT avoir des transitions originales entres les parties et "
                            "sous-parties. Ces transitions doivent annoncer la suite du texte d'une manière enthousiaste",
                            "L'article final doit ABSOLUMENT être très détaillé dans les explications. La tonalité doit "
                            "être légèrement formelle, mais accessible et enthousiaste",
                        ]
                    ]
                    prompt_reformulate_content = [
                        generate_prompt_reformulate_content(
                            st.session_state.keyword,
                            st.session_state["selected_structure"],
                            exs,
                            len_content,
                            st.session_state["intention_recherche"],
                            st.session_state["keywords_list"],
                            st.session_state["related_questions"],
                            st.session_state["maillage_interne"],
                            a,
                            s,
                        )
                        for a, s in zip(draft_content_proposals, supplement_prompt)
                    ]

                    # => Rephrase & Final content
                    st.session_state.content_suggests = [
                        st.session_state["llm"].invoke(p)
                        for p in prompt_reformulate_content
                    ]
                    if len(st.session_state.content_suggests) < 3:
                        st.error(
                            "Erreur dans la génération de contenu, Veuillez réessayer"
                        )
                        st.session_state["content_ready"] = False
                    else:
                        st.session_state["content_ready"] = True
                except:
                    st.error("Erreur dans la génération de contenu, Veuillez réessayer")
                    st.session_state["content_ready"] = False

                reset_generation(step="content")

            end = time.time()
            response_time_sec = end - start
            st.success(f"Contenus générés en {int(response_time_sec)} secondes")

    # Affichage résultat
    if st.session_state["content_ready"]:

        content_suggests = st.session_state.content_suggests
        keywords_list = [s.strip() for s in mots_cles.split(",")]

        def afficher_proposition(proposition_num, content, keywords_list):

            sem_score = calculate_semantic_score(content, keywords_list)
            len_text = len(content.split())
            expand = (
                (st.session_state["n_content_choix"] == str(proposition_num))
                if st.session_state["n_content_choix"]
                else False
            )

            with st.expander(
                f"Proposition {proposition_num} | Score sémantique : {sem_score} | Nb mots ≃ {len_text}",
                expanded=expand,
            ):
                # Utilisation de st.text_area pour rendre le contenu éditable
                edited_content = st.text_area(
                    f"Proposition {proposition_num}",
                    value=content.replace("```markdown\n", "").replace("\n```", ""),
                    height=500,
                )

                # Bouton d'actualisation du score
                actualiser_score_btn = st.button(
                    f"Actualiser le score de la proposition {proposition_num}",
                    key=f"actualiser_score_{proposition_num}_btn",
                )

                # Si le bouton est cliqué, recalculer le score et la longueur du texte
                if actualiser_score_btn:
                    # Recalculer les nouvelles valeurs avec le contenu édité
                    new_len_text = len(edited_content.split())
                    new_sem_score = calculate_semantic_score(
                        edited_content, keywords_list
                    )

                    # Afficher les nouvelles valeurs recalculées
                    st.write(f"Nouvelle longueur du texte : {new_len_text} mots")
                    st.write(f"Nouveau score sémantique : {new_sem_score}")

                # Bouton de sélection
                prop_content_btn = st.button(
                    f"Choisir cette proposition",
                    key=f"prop_content_{proposition_num}_btn",
                    use_container_width=True,
                )

                if prop_content_btn:
                    # Utiliser le contenu édité
                    selected_content = add_hn_label(edited_content)
                    st.session_state["selected_content"] = selected_content
                    st.session_state["n_content_choix"] = str(proposition_num)
                    reset_generation(step="content")

        # Affichage des colonnes et des propositions
        c_c1, c_c2, c_c3 = st.columns(3)

        # Proposition 1
        with c_c1:
            afficher_proposition(1, content_suggests[0], keywords_list)

        # Proposition 2
        with c_c2:
            afficher_proposition(2, content_suggests[1], keywords_list)

        # Proposition 3
        with c_c3:
            afficher_proposition(3, content_suggests[2], keywords_list)

        if len(st.session_state["selected_content"]) > 1:
            txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Proposition de contenu 
                        choisie: {str(st.session_state["n_content_choix"])} </p>"""
            st.markdown(txt, unsafe_allow_html=True)

    # *********************************************************
    # ===> Proposition de titre
    # *********************************************************
    st.divider()
    generate_title_btn = st.button(
        "Générer un titre et une description", key="generate_title"
    )

    # ****===> Génération
    if generate_title_btn:
        txt = f"""<p style="color:{font_col_txt}; font-size: 30px;">Génération de titre et description </p>"""
        st.markdown(txt, unsafe_allow_html=True)
        # Test if data available or if keyword changed
        if (len(st.session_state["concurrents_data"]) < 1) or (
            st.session_state.keyword != keyword
        ):
            parsed_res = get_data()
        concurrents = st.session_state["concurrents_data"]
        concurrents_title = [i["title"] for i in concurrents]
        concurrents_desc = [i["descr"] for i in concurrents]

        exs = "\n".join(
            [
                f"## Début exemple \n Titre : {t} | Description : {d} \n## Fin exemple \n"
                for t, d in zip(concurrents_title, concurrents_desc)
            ]
        )

        with st.spinner(
            "Génération des propositions de titre & description. Le traitement peut prendre quelques "
            "secondes ..."
        ):
            start = time.time()
            chain_title = build_chain(st.session_state["llm"], Title)
            prompt_title = generate_prompt_title(
                st.session_state.keyword, st.session_state["selected_content"], exs
            )

            try:
                res_title_llm = chain_title.invoke({"query": prompt_title})
                st.session_state.title_suggests = res_title_llm.title

                if len(res_title_llm.title) < 3:
                    st.error("Erreur dans la génération du titre, veuillez réessayer")
                    st.session_state["title_ready"] = False
                else:
                    st.session_state["title_ready"] = True
            except:
                st.error("Erreur dans la génération du titre, veuillez réessayer")
                st.session_state["title_ready"] = False

            st.session_state["n_title_choix"] = None

        end = time.time()
        response_time_sec = end - start
        st.success(f"Titre et description générés en {int(response_time_sec)} secondes")

    # => Affichage titre
    if st.session_state["title_ready"]:
        title_suggests = st.session_state.title_suggests

        c_c1, c_c2, c_c3 = st.columns(3)
        with c_c1:
            with st.expander("Proposition 1", expanded=True):
                key = list(title_suggests.keys())[0]
                title = title_suggests[key][0]
                desc = title_suggests[key][1]

                st.text_area(
                    "Titre",
                    f""" Titre : {title} \n\n Description : {desc}""",
                    height=200,
                )
                prop_title_1_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_title_1_btn",
                    use_container_width=True,
                )
                if prop_title_1_btn:
                    st.session_state["selected_title"] = title_suggests[key]
                    st.session_state["n_title_choix"] = "1"

        with c_c2:
            with st.expander("Proposition 2", expanded=True):
                key = list(title_suggests.keys())[1]
                title = title_suggests[key][0]
                desc = title_suggests[key][1]

                st.text_area(
                    "Titre",
                    f""" Titre : {title} \n\n Description : {desc}""",
                    height=200,
                )
                prop_title_2_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_title_2_btn",
                    use_container_width=True,
                )
                if prop_title_2_btn:
                    st.session_state["selected_title"] = title_suggests[key]
                    st.session_state["n_title_choix"] = "2"

        with c_c3:
            with st.expander("Proposition 3", expanded=True):
                key = list(title_suggests.keys())[2]
                title = title_suggests[key][0]
                desc = title_suggests[key][1]

                st.text_area(
                    "Titre",
                    f""" Titre : {title} \n\n Description : {desc}""",
                    height=200,
                )
                prop_title_3_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_title_3_btn",
                    use_container_width=True,
                )
                if prop_title_3_btn:
                    st.session_state["selected_title"] = title_suggests[key]
                    st.session_state["n_title_choix"] = "3"
    if st.session_state["n_title_choix"]:
        txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Proposition de titre/description 
                choisie: {str(st.session_state["n_title_choix"])} </p>"""
        st.markdown(txt, unsafe_allow_html=True)

    # *********************************************************
    # ===> Proposition de résumé
    # *********************************************************
    st.divider()
    content_input = st.text_area(
        "Contenu à résumer",
        value="",
        placeholder="Renseignez votre texte à résumer",
        height=400,
    )
    generate_summary_btn = st.button("Générer un résumé", key="generate_summary")

    if generate_summary_btn:
        # Test if data available or if keyword changed
        if (len(st.session_state["concurrents_data"]) < 1) or (
            st.session_state.keyword != keyword
        ):
            parsed_res = get_data()
        if len(content_input) < 20:
            st.warning("Le résumé sera généré sans contenu...")
        txt = f"""<p style="color:{font_col_txt}; font-size: 30px;">Génération de résumé </p>"""
        st.markdown(txt, unsafe_allow_html=True)

        # ****===> Génération
        with st.spinner(
            "Génération des propositions de résumé. Le traitement peut prendre quelques secondes ..."
        ):
            start = time.time()
            chain_summary = build_chain(st.session_state["llm"], Summary)

            prompt_summary = generate_prompt_summary(
                subject=st.session_state.keyword,
                content=content_input,
                intention=st.session_state["intention_recherche"],
                list_ancre=st.session_state["maillage_interne"],
                len_summary=MAX_LEN_SUMMARY,
            )

            st.session_state.summary_suggests = get_summary(
                prompt_summary=prompt_summary, chain_summary=chain_summary, st=st
            )
            st.session_state["n_summary_choix"] = None

            end = time.time()
            response_time_sec = end - start
            st.success(f"Résumés générés en {int(response_time_sec)} secondes")

    # => Affichage Résumés
    if st.session_state["summary_ready"]:
        summary_suggests = st.session_state.summary_suggests

        c_c1, c_c2, c_c3 = st.columns(3)
        with c_c1:
            with st.expander("Proposition 1", expanded=True):
                summary = summary_suggests[0]

                st.text_area("Résumé", summary, height=500)
                prop_summary_1_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_summary_1_btn",
                    use_container_width=True,
                )
                if prop_summary_1_btn:
                    st.session_state["selected_summary"] = summary
                    st.session_state["n_summary_choix"] = "1"

        with c_c2:
            with st.expander("Proposition 2", expanded=True):
                summary = summary_suggests[1]

                st.text_area("Résumé", summary, height=500)
                prop_summary_2_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_summary_2_btn",
                    use_container_width=True,
                )
                if prop_summary_2_btn:
                    st.session_state["selected_summary"] = summary
                    st.session_state["n_summary_choix"] = "2"

        with c_c3:
            with st.expander("Proposition 3", expanded=True):
                summary = summary_suggests[2]

                st.text_area("Résumé", summary, height=500)
                prop_summary_3_btn = st.button(
                    "Choisir cette proposition",
                    key="prop_summary_3_btn",
                    use_container_width=True,
                )
                if prop_summary_3_btn:
                    st.session_state["selected_summary"] = summary
                    st.session_state["n_summary_choix"] = "3"

    if st.session_state["n_summary_choix"]:
        txt = f"""<p style="color:{font_col_txt}; font-size: 15px;">Proposition de résumé 
                choisie: {str(st.session_state["n_summary_choix"])} </p>"""
        st.markdown(txt, unsafe_allow_html=True)

    # => Afficher récapitulatif
    if (
        st.session_state["n_structure_choix"]
        and st.session_state["n_content_choix"]
        and st.session_state["n_title_choix"]
        and st.session_state["n_summary_choix"]
    ):
        st.divider()
        txt = f"""<p style="color:{font_col_txt}; font-size: 30px;">Récapitulatif des choix </p>"""
        st.markdown(txt, unsafe_allow_html=True)
        with st.expander("Afficher les choix finaux", expanded=True):
            tab1_choix_fin, tab2_choix_fin, tab3_choix_fin, tab4_choix_fin = st.tabs(
                ["Titre et description", "Structure HN", "Article", "Résumé"]
            )

            with tab1_choix_fin:
                title = st.session_state["selected_title"][0]
                desc = st.session_state["selected_title"][1]
                st.text_area(
                    "Titre-Description",
                    f""" Titre : {title} \n\n Description : {desc}""",
                    height=200,
                    key="recap_title",
                )

            with tab2_choix_fin:
                structure = st.session_state["selected_structure"]
                st.text_area(
                    "Structure Hn",
                    html_to_markdown_with_levels(hn_list_to_md(structure)),
                    height=450,
                    key="recap_structure",
                )

            with tab3_choix_fin:
                article = st.session_state["selected_content"]
                st.markdown(article)

            with tab4_choix_fin:
                summary = st.session_state["selected_summary"]
                st.markdown(summary)


if __name__ == "__main__":
    main()
