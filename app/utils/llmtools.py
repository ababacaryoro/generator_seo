import json
import os
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import yaml
from yaml.loader import SafeLoader

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "banned_expression.yml")) as f:
    banned_expr_list = yaml.load(f, Loader=SafeLoader)
banned_expr_list = [s.strip() for s in banned_expr_list["expressions"]]


class StructureHn(BaseModel):
    structure: dict[str, list[list[str, str]]] = Field(
        description="Dictionnaire de 3 propositions de structure Hn. Ceux-ci sont au format de liste avec 2 éléments "
        ": le niveau (h1,h2, etc) et le titre"
    )


class Content(BaseModel):
    content: dict[str, list[list[str, str, str]]] = Field(
        description="Dictionnaire de 3 propositions de contenu pour une structure Hn. Chaque proposition est au "
        "format de liste avec 3 éléments : le niveau (h1,h2, etc), le titre et le contenu"
    )


class PartContent(BaseModel):
    content: list[str, str, str] = Field(
        description="Liste de 3 propositions de contenu pour une partie de l'article de blog. Chaque proposition est "
        "un string"
    )


class Title(BaseModel):
    title: dict[str, list[str, str]] = Field(
        description="Dictionnaire de 3 propositions de titre avec méta-description. Chaque proposition est au "
        "format de liste avec 2 éléments : le titre et la méta-description"
    )


class Summary(BaseModel):
    summary: list[str, str, str] = Field(
        description="Dictionnaire de 3 propositions de résumé. Chaque proposition est au "
        "format de string"
    )


def build_chain(llm, output_class):
    parser = PydanticOutputParser(pydantic_object=output_class)

    prompt = PromptTemplate(
        template="Answer the user query based on the information given.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    return chain


def get_intention_recherche(keyword, llm):
    question = f"""Quelle est l'intention de recherche d'une personne tapant dans Google le mot-clé : "{keyword}"? 
    Que veut-elle savoir, retrouver dans les contenus qui vont apparaitre ?"""
    intention: str = llm.invoke(question)

    return intention


def get_init_structure(keyword, intention_recherche, llm):
    question = f"""Nous devons rédiger un article de blog sur la thématique "{keyword}". Tu as ci-dessous un descriptif 
    de l'intention de recherche d'une personne sur ce sujet : ce qu'il aimerait savoir. Sur la base de ces informations, 
    quelle structure HN aurais-tu proposé  pour un article de blog ? Donne la structure HN en markdown. Le style et le 
    ton de la structure doivent correspondre à ceux des articles de blog.
    ## Intention de recherche : 
    "{intention_recherche}"
    Structure Hn : """
    init_structure: str = llm.invoke(question)

    return init_structure


def generate_prompt_structure(subject, examples, related_questions, init_structure):
    prompt = f"""
    Vous êtes un employé SEO de Docaposte. Vous devez écrire un article de blog pour le site web de Docaposte, sur un 
    sujet donné. Pour cela, vous disposez d'un premier brouillon de structure Hn que vous devez améliorer en se basant 
    sur les éléments suivants : 
        - Plusieurs exemples de structure Hn de pages bien référencées, parlant du sujet. 
        - Des questions fréquentes que se posent les internautes sur le sujet.
    Ces informations doivent vous permettre de proposer 3 version améliorées de structure Hn pour l'article de blog à 
    rédiger. Les 3 propositions doivent se différencier sur certains points thématiques abordés.
    Votre réponse doit être uniquement un dictionnaire de 3 listes de listes Python, qui donnent les 3 structures Hn à
    utiliser pour le contenu. Votre réponse doit être comme dans les exemples ci-dessous, sans aucun commentaire.
    
    ### Entrées :
    
    1. **Sujet de l'article à rédiger** :
    {subject}
    
    2. **Premier brouillon de structure Hn** :
    {init_structure}
    
    3. **Exemples structures Hn des pages les mieux référencées** :
    {examples}
    
    4. **Autres questions fréquentes que se posent les internautes sur le sujet** :
    {related_questions}
    
    
    ### Sortie attendue :
    
    Un dictionnaire Python contenant trois propositions de structure Hn. Chaque proposition doit être sous forme de 
    liste ayant des sous listes composés de 2 éléments sous ce format : ["h1", "le titre"].
    
    Exemple de sortie :
        {{
            "proposition1": [
                                    ["h1", "Mon titre 1"], 
                                    ["h2", "Mon sous-titre 1"], 
                                    ["h2", "Mon sous-titre 2"], 
                                    ["h1", "Mon titre 2"], 
                                    ["h1", "Mon titre 3"]
            ],
            "proposition2": [
                                    ["h1", "Mon titre 1"], 
                                    ["h2", "Mon sous-titre 1"], 
                                    ["h2", "Mon sous-titre 2"], 
                                    ["h1", "Mon titre 2"], 
                                    ["h1", "Mon titre 3"]
            ],
            "proposition3": [
                                    ["h1", "Mon titre 1"], 
                                    ["h2", "Mon sous-titre 1"], 
                                    ["h2", "Mon sous-titre 2"], 
                                    ["h1", "Mon titre 2"], 
                                    ["h1", "Mon titre 3"]
            ]
        }}
        
    ### Instructions :
    
    1. S'inspirer des exemples de structure HN fournis pour trouver plusieurs thématiques à aborder. Les 3 propositions
    devront avoir au moins quelques différences sur les thématiques abordées.
    2. Utiliser des guillemets doubles pour tout texte dans le résultat.
    3. Ne pas proposer un seul sous-niveau de titre Hn inférieur en dessous d'un niveau H(n+1) : par exemple, ne pas 
    proposer un seul H3 sous un H2.
    4. Ne proposer qu'un seul niveau H1 qui présente le sujet de l'article de blog. Sa formulation doit être captivante, 
    adaptée pour un article de blog.
    5. Au moins une des autres questions fréquentes que se posent les internautes sur le sujet doit être pris en 
    compte dans la structure proposée.
    6. Dans la formulation des phrases, utiliser un ton adapté pour des articles de blog. Ne pas hésiter à s'adresser au
    lecteur en le vouvoyant ou en parlant pour le compte de Docaposte. 
    7. Le ton doit être également rassurant, expert et pédagogique, qui vulgarise le contenu technique tout en ajoutant 
    des éléments de langage "expert" adaptés au sujet.
    8. Il ne faut pas se limiter à changer la formulation ou l'agencement du texte pour différencier les 3 propositions. 
    Chacune des propositions doit avoir des parties qui lui sont propres, en se basant sur ce qui existe dans 
    les exemples fournis et les autres questions fréquentes posées par les internautes.
    9. Éviter de nommer des solutions ou entreprises concurrentes à Docaposte.
    
    ### Sortie : 
     
    """
    return prompt


def generate_structure_proposals(
    chain, subject, examples, related_questions, init_structure, n_proposals=3
):
    """
    Relancer la génération des structures tant qu'on a pas 3 propositions
    Args:
        chain:
        subject:
        examples:
        related_questions:
        init_structure:
        n_proposals:

    Returns:

    """
    proposals = {}
    counter = 0
    max_iter = 15
    while (counter < max_iter) and (len(proposals) < n_proposals):
        prompt_structure = generate_prompt_structure(
            subject, examples, related_questions, init_structure
        )
        try:
            print("-- Start structure generation")
            res = chain.invoke({"query": prompt_structure})
            print("-- Structure generated")
            new_proposal = res.structure
            if len(new_proposal) == n_proposals:
                proposals = new_proposal
        except:
            print("-- Failed to generate")
            pass

        counter += 1
    # Completer par des chaines vides non-reussite
    if len(proposals) < n_proposals:
        print(f"==> Retour de structures vides pour completer")

    print(f"==> Did {counter} iterations to generate proposals")

    return proposals


def generate_prompt_part_content(
    subject,
    structure_hn,
    examples,
    intention_recherche,
    mots_cles,
    related_questions,
    article_start="",
    part_cible="",
):
    """
    Générer une partie du contenu en ayant le début de l'article
    Args:
        subject:
        structure_hn:
        examples:
        intention_recherche:
        mots_cles:
        related_questions:
        article_start:
        part_cible:

    Returns:

    """

    prompt = f"""
    Vous êtes un employé SEO de Docaposte. Vous devez participer à la rédaction d'un article de blog pour le site 
    web de Docaposte, sur un sujet donné. Vous avez à votre charge la rédaction d'une seule section de l'article.
    Pour cela, vous disposez d'une structure HN que l'article doit respecter. Vous avez aussi 3 versions du début 
    de l'article déjà rédigé et vous devez fournir 3 propositions de contenu pour une section. 
    Tous les éléments en entrée sont les suivants :
        - le sujet de l'article de blog à rédiger;
        - l'intention de recherche des internautes qui se posent des questions sur ce sujet;
        - les autres questions fréquentes que se posent les internautes sur le sujet;
        - les mots clés important à faire figurer dans le contenu pour un meilleur référencement de l'article sur google;
        - la structure HN à respecter, sous forme d'une liste avec des sous-listes du type ["h1", "titre de la section"];
        - des exemples de contenus d'articles, parlant du sujet;
        - 3 versions pour le début de l'article déjà rédigé;
        - La partie de l'article à rédiger.
    Ces informations doivent vous permettre de fournir 3 propositions de contenu pour une section de l'article de blog à 
    rédiger, adaptées aux 3 versions du début de l'article déjà rédigées. Les 3 propositions doivent se différencier et 
    être des suites logiques aux 3 versions de début de l'article déjà rédigées.

    ### Entrées :

    1. **Sujet de l'article à rédiger** :
        {subject}

    2. **Intention de recherche des utilisateurs à travers ce sujet** :
        {intention_recherche}

    3. **Autres questions similaires que se posent les internautes sur le sujet** :
        {related_questions}

    4. **Mots clés importants pour l'article à rédiger** :
        {mots_cles}

    4. **Structure Hn à respecter** :
        {structure_hn}

    5. **Exemples d'articles** :
        {examples}

    6. **3 versions de début de l'article déjà rédigé** :
        {article_start}

    7. **Partie de l'article à rédiger** :
        {part_cible}

    ### Sortie attendue :

    Une liste Python contenant trois propositions de texte pour la partie correspondante de l'article, chaque 
    proposition doit être de type string, contenant un texte cohérent avec le début d'article fourni.

    Exemple de sortie :
    [
     "Contenu généré pour la proposition 1",
     "Contenu généré pour la proposition 2",
     "Contenu généré pour la proposition 3",
    ]

    ### Instructions :

    1. Respecter strictement le format de sortie avec une liste Python de 3 textes correspondant aux propositions de 
    contenu.
    2. Dans la formulation des phrases, utiliser un ton adapté pour des articles de blog. S'adresser au
    lecteur en le vouvoyant et en parlant pour le compte de Docaposte. 
    3. Le ton doit être également rassurant, expert et pédagogique, qui vulgarise le contenu technique tout en ajoutant 
    des éléments de langage "expert" adaptés au sujet.
    4. Utiliser des guillemets doubles pour tout texte dans le résultat.
    5. Différencier les 3 propositions en donnant une qui propose plusieurs exemples avec pédagogie, une autre qui 
    contient des transitions originales entre les parties et une dernière qui soit très détaillée.
    6. Éviter de nommer des solutions ou entreprises concurrentes à Docaposte.
    7. Détailler le contenu avec des explications illustrées au besoin.
    8. Ne pas empiéter sur une autre partie de la structure HN. Se limiter à donner des informations spécifiques à la 
    partie cible de l'article à rédiger.
    9. Il faut obligatoirement en réponse une liste Python de 3 textes.
    10. Il faut nécessairement que les 3 propositions soient des suites logiques des 3 versions de début de l'article 
    fournies. 
    11. Les 3 propositions doivent ABSOLUMENT avoir un style de rédaction différent : 
        - La première doit contenir plusieurs exemples.
        - la deuxième doit avoir des transitions originales entres les parties.
        - la troisième doit être très détaillée dans les explications avec une tonalité légèrement formelle, mais 
        accessible et enthousiaste.
    12. Le contenu généré doit avoir une approche pyramidale, en commençant par les éléments les plus généraux pour 
    ensuite aller progressivement vers des détails plus spécifiques.
    
    ### Réponse : 
    """
    return prompt


def get_part_content_proposals(
    chain,
    subject,
    structure_hn,
    examples,
    intention_recherche,
    mots_cles,
    related_questions,
    article_start,
    part_cible,
):
    """
    Relancer la génération du contenu tant qu'il n'y a pas 3 propositions

    Args:
        chain:
        subject:
        structure_hn:
        examples:
        intention_recherche:
        mots_cles:
        related_questions:
        article_start:
        part_cible:

    Returns:

    """
    proposals = []
    counter = 0
    max_iter = 15
    n_proposals = 3
    while (counter < max_iter) and (len(proposals) < n_proposals):
        prompt_part_content = generate_prompt_part_content(
            subject,
            structure_hn,
            examples,
            intention_recherche,
            mots_cles,
            related_questions,
            article_start,
            part_cible,
        )
        try:
            print("-- Start generation")
            res = chain.invoke({"query": prompt_part_content})
            print("-- Content generated")
            new_proposal = res.content
            if len(new_proposal) == n_proposals:
                proposals = new_proposal
            else:
                proposals += new_proposal
        except:
            print("-- Failed to generate")
            pass

        counter += 1
    # Completer par des chaines vides non-reussite
    if len(proposals) < n_proposals:
        print(
            f"==> Rajout de {n_proposals - len(proposals)} chaines vides pour completer"
        )
        while len(proposals) < n_proposals:
            proposals.append("")

    print(f"==> Did {counter} iterations to generate proposals")

    return proposals


def get_summary(prompt_summary, chain_summary, st):
    """loop to get the format"""
    summary_len = 0
    counter = 0
    max_iter = 25
    while (counter < max_iter) & (summary_len < 3):
        try:
            print("-- Start summary generation")
            res_summary_llm = chain_summary.invoke({"query": prompt_summary})
            # res_summary_llm = st.session_state.llm.invoke(prompt_summary)
            # print(res_summary_llm)
            summary_len = len(res_summary_llm.summary)
            st.session_state.summary_suggests = res_summary_llm.summary
            print(f"-- counter {counter} summary_len  {summary_len}")

        except:
            print(f"-- Failed to generate {counter}")
            st.session_state["summary_ready"] = False
            pass

        counter += 1

    if len(st.session_state.summary_suggests) < 3:
        st.error("Erreur dans la génération du résumé, veuillez réessayer")
        st.session_state["summary_ready"] = False
    else:
        st.session_state["summary_ready"] = True

    return st.session_state.summary_suggests


def get_all_content_proposals(
    chain,
    subject,
    structure_hn,
    examples,
    intention_recherche,
    mots_cles,
    related_questions,
):
    """
    Générer tout le contenu de manière séquentielle

    Args:
        chain:
        subject:
        structure_hn:
        examples:
        intention_recherche:
        mots_cles:
        related_questions:

    Returns:

    """

    articles = ": ".join((structure_hn[0]))
    articles = [articles, articles, articles]
    for i in range(len(structure_hn)):
        if i == 0:
            continue

        part_cible = ": ".join((structure_hn[i]))
        debut_art = "\n\n---\n".join(
            [
                "Version " + str(a + 1) + ": \n\n" + articles[a]
                for a in range(len(articles))
            ]
        )
        proposals = get_part_content_proposals(
            chain,
            subject,
            structure_hn,
            examples,
            intention_recherche,
            mots_cles,
            related_questions,
            debut_art,
            part_cible,
        )
        articles = [
            art + "\n\n" + part_cible + "\n\n" + prop
            for art, prop in zip(articles, proposals)
        ]

    return articles


def generate_prompt_reformulate_content(
    subject,
    structure_hn,
    examples,
    len_content,
    intention_recherche,
    mots_cles,
    related_questions,
    list_ancre="",
    article_init="",
    supplement_prompt="",
    banned_expr=", ".join(banned_expr_list),
):
    """
    Reformuler l'article déjà généré
    Args:
        banned_expr:
        supplement_prompt:
        subject:
        structure_hn:
        examples:
        len_content:
        intention_recherche:
        mots_cles:
        related_questions:
        list_ancre:
        article_init:

    Returns:

    """
    prompt = f"""
    Vous êtes un employé SEO de Docaposte. Vous devez écrire un article de blog pour le site web de Docaposte, sur un 
    sujet donné. Pour cela, vous disposez d'une structure HN que l'article doit respecter et d'un premier brouillon de 
    contenu d'article. Vous devez utiliser ces inputs pour proposer une nouvelle version reformulée et résumée de 
    l'article, cohérent, avec des transitions entre les différentes parties. Pour cela, vous disposez également des 
    éléments complémentaires suivants:
        - le sujet de l'article de blog à rédiger;
        - l'intention de recherche des internautes qui se posent des questions sur ce sujet;
        - les autres questions fréquentes que se posent les internautes sur le sujet;
        - les mots clés important à faire figurer dans le contenu pour un meilleur référencement de l'article sur google;
        - la structure HN à respecter, sous forme d'une liste avec des sous-listes du type ["h1", "titre de la section"];
        - des titres et liens d'articles similaires au sujet et disponible sur le site de Docaposte ;
        - des exemples de contenus d'articles, parlant du sujet.

    ### Entrées :

    1. **Sujet de l'article à rédiger** :
        {subject}

    2. **Intention de recherche des utilisateurs à travers ce sujet** :
        {intention_recherche}

    3. **Autres questions similaires que se posent les internautes sur le sujet** :
        {related_questions}

    4. **Mots clés importants pour l'article à rédiger** :
        {mots_cles}

    5. **Structure Hn à respecter** :
        {structure_hn}

    6. **Titre des thématiques similaires abordées dans des pages de Docaposte ainsi que les liens correspondants** :
        {list_ancre}

    7. **Exemples d'articles** :
        {examples}

    8. **Premiere version de l'article de blog** :
        {article_init}

    ### Sortie attendue :

    L'article de blog reformulé et respectant la structure HN fournie en entrée ainsi que le format de la première 
    version. Le texte doit être écrit en format markdown.

    ### Instructions :

    1. Reformuler le brouillon fourni en donnant un texte final contenant environ {len_content} mots et en 
    respecter strictement le format de sortie demandé. Utiliser au maximum les mots clés donnés pour le texte reformulé.
    2. Dans la formulation des phrases, utiliser un ton adapté pour des articles de blog. S'adresser au
    lecteur en le vouvoyant et en parlant pour le compte de Docaposte. 
    3. Mettre un paragraphe d'introduction à l'article de blog après le H1 qui correspond au titre. Les autres titres de 
    section également devront OBLIGATOIREMENT avoir un paragraphe introduction pour parler brièvement du contenu de la section.
    4. Le ton doit être également rassurant, expert et pédagogique, qui vulgarise le contenu technique tout en ajoutant 
    des éléments de langage "expert" adaptés au sujet.
    5. Utiliser des guillemets doubles pour tout texte dans le résultat.
    6. Éviter de nommer des solutions ou entreprises concurrentes à Docaposte.
    7. Détailler le contenu mais éliminer les répétitions.
    8. Le début des paragraphes doit être varié et original. Eviter d'utiliser tout le temps le mot "Pour" dans le 
    début des paragraphes.
    9. Éviter ABSOLUMENT d'utiliser les expressions suivantes : {banned_expr}.
    10. Ne jamais utiliser d'adjectifs verbaux (verbes qui se terminent en "ant")
    11. Le contenu généré doit avoir une approche pyramidale, en commençant par les éléments les plus généraux pour 
    ensuite aller progressivement vers des détails plus spécifiques.
    12. La conclusion devrait être sous forme d'ouverture, avec une question stimulante ou l'orientation de la réflexion
     vers d'autres aspects de la thématique, incitant ainsi à poursuivre la réflexion.
    13. Vous devez OBLIGATOIREMENT insérer dans le texte tous les liens vers les thématiques similaires abordées 
    dans les pages de Docaposte, donnés en entrée ci-haut. Chaque lien devra être inséré une seule fois. Pour cela, 
    quand un mot ou groupe de mots présent dans le titre d'une des thématiques similaires fournies apparait dans le 
    brouillon d'article, dans la reformulation, mettre immédiatement après ce mot ou groupe de mots le lien 
    correspondant à cette thématique entre des parenthèses. Par exemple, si la thématique fournie similaire est 
    "Titre : 'signature électronique' | Lien : https://www.docaposte.com/solutions/signature-electronique", et 
    que le brouillon d'article contient "Pour sécuriser vos documents, une signature électronique est essentielle...",
    vous pourrez mettre dans le texte reformulé un passage comme suit : 
    "Pour sécuriser vos documents, une signature électronique (https://www.docaposte.com/solutions/signature-electronique) est essentielle...".
    Il faut nécessairement que le lien inséré entre parenthèses soit après un mot ou groupe de mots présent dans le 
    titre de la thématique similaire. 
    14. Chaque lien vers les thématiques similaires abordées dans les pages de Docaposte ne doit être inséré qu'une 
    seule fois dans le texte et l'insertion devra être naturelle, subtile, en évitant les expressions du type : 
    "Pour en savoir plus, consulter cette page (lien) ...". Il faut nécessairement insérer tous les liens dans le texte.
    
    
    ### Article final :
    {supplement_prompt}
    
    """
    return prompt


def generate_prompt_title(subject, content, examples):
    prompt = f"""
    Vous êtes un employé SEO de Docaposte. Vous devez écrire un article de blog pour le site web de Docaposte, sur un 
    sujet donné. Vous venez de finir la rédaction et on vous demande de proposer un titre et une méta-description pour
    la page web de cet article. Pour cela, vous disposez de l'article complet, des titres et méta-descriptions d'autres 
    articles similaires parlant du même sujet et de la thématique générale de l'article.
    Votre objectif est maintenant de proposer 3 combinaisons de titre et méta-description sous forme d'un dictionnaire 
    contenant des listes. Les titres et méta-description devront être originaux et adaptés pour un article de blog.
    

    ### Entrées :

    1. **Sujet de l'article** :
        {subject}

    2. **Contenu de l'article de blog** :
        {content}

    3. **Exemples de titres et méta-descriptions de pages existantes** :
        {examples}

    ### Sortie attendue :

    Un dictionnaire Python contenant trois propositions de titre et méta-description, chaque proposition mis sous format
    de liste. Ces listes doivent contenir le titre et la méta-description, comme suit: 
    ["Mon titre", "Ma métadescription"].

    Exemple de sortie :
    {{
        "proposition1": ["Titre", "méta-description"],
        "proposition2": ["Titre", "méta-description"],
        "proposition3": ["Titre", "méta-description"]
    }}

    ### Instructions :

    1. Donnez 3 propositions de titre et méta-description et respecter strictement le format de sortie.
    2. S'inspirer des exemples donnés pour le ton, le style et la taille des titres et méta-descriptions. Le style doit
    être adapté à un article de blog.
    3. le titre doit comporter au maximum 60 caractères et la description doit avoir au maximum 160 caractères. 
    4. le sujet de l'article doit être inclus à la fois dans le titre et la description.
    5. Les 3 propositions de description doivent être différenciées en tenant compte des contraintes suivantes :
        - Une proposition avec appel à l’action ainsi qu'une mise en avant d’avantages. Par exemple: « Découvrez les étapes pour faire une signature électronique en quelques clics. ✓ Tous types de fichiers ✓ 100% français »...).
        - Une proposition avec un début de réponse. Par exemple, pour le sujet « valeur juridique signature électronique », certaine version pourrait être : « La valeur juridique de la signature électronique est encadrée par le règlement eIDAS et le Code civil français pour assurer la validité de vos documents... »).
        - Une proposition commençant par une question. Par exemple : « Vous vous demandez comment faire une signature électronique facilement ? Suivez nos étapes simples pour sécuriser vos documents et rester conforme à la législation... »).
    6. Les 3 propositions de titre doivent être différenciées en tenant compte des contraintes suivantes :
        - Une proposition sous forme de question
        - Une autre proposition commençant par le sujet de l'article suivi de ":" et une expression ou une phrase. Par exemple : « Faire une signature électronique : tout ce qu'il vous faut pour être à la page»).
    
    ### Sortie : 
    """
    return prompt


def generate_prompt_summary(
    subject,
    content,
    intention,
    list_ancre,
    len_summary=200,
    banned_expr=", ".join(banned_expr_list),
):
    if len(content) < 1:
        content = "inconnu"
    prompt = f"""
    Vous êtes un employé SEO de Docaposte. Vous devez écrire un article de blog pour le site web de Docaposte, sur un 
    sujet donné. Vous venez de finir la rédaction et on vous demande de proposer un résumé pour cet article. Pour cela, 
    vous disposez de l'article complet et du sujet de l'article. 
    Votre objectif est maintenant de proposer 3 versions de résumés sous forme d'un dictionnaire contenant 3 textes. 
    Les résumés devront être originaux et adaptés pour un article de blog.


    ### Entrées :

    1. **Sujet de l'article** :
        {subject}

    2. **Intention de recherche des internaute lisant l'article** :
        {intention if intention is not None else 'inconnu'}
    
    3. **Contenu de l'article de blog** :
        {content if content is not None else 'inconnu'}

    4. **Titre des thématiques similaires abordées dans des pages de Docaposte ainsi que les liens correspondants** :
        {list_ancre}
        
    ### Sortie attendue :

    Un dictionnaire Python contenant trois propositions de résumé, chaque proposition étant un texte. 
    
    Exemple de sortie :
    [
        "Contenu pour le résumé 1",
        "Contenu pour le résumé 2",
        "Contenu pour le résumé 3"
    ]

    ### Instructions :

    1. Donnez 3 propositions de résumé et respecter strictement le format de sortie.
    2. Dans la formulation des phrases, utiliser un ton adapté pour des articles de blog. S'adresser au
    lecteur en le vouvoyant et en parlant pour le compte de Docaposte. 
    3. Le ton doit être également rassurant, expert et pédagogique, qui vulgarise le contenu technique tout en ajoutant 
    des éléments de langage "expert" adaptés au sujet.
    4. Faire un résumé de tous les principaux points évoqués dans le contenu.
    5. Le résumé doi t ABSOLUMENT avoir au maximum {len_summary} mots.
    6. Éviter ABSOLUMENT éviter d'utiliser les expressions suivantes : {banned_expr}. 
    7. Ne jamais utiliser d'adjectifs verbaux (verbes qui se terminent en "ant") 
    8. Ne jamais inclure d’éléments promotionnel, par exemple "Docaposte vous accompagne dans…". 
    9. Chacun des résumés doit ABSOLUMENT inclure 2 ou 3 listes à puces pour lister les principaux éléments du contenu. 
    10. Chacun des résumés doit ABSOLUMENT reprendre les éléments de l’article les plus importants en lien avec 
    l’intention de recherche lorsque ces informations sont connues. 
    11. Éviter de nommer des solutions ou entreprises concurrentes à Docaposte. 
    12. Vous devez OBLIGATOIREMENT insérer dans le texte tous les liens vers les thématiques similaires abordées dans 
    les pages de Docaposte, donnés en entrée ci-haut. Chaque lien devra être 
    inséré une seule fois. Pour cela, détecter dans le contenu un mot ou groupe de mots présent dans le titre d'une des 
    thématiques similaires fournies apparait dans le texte, mettre immédiatement après ce mot ou groupe de mots le lien 
    correspondant à cette thématique entre des parenthèses. Par exemple, si la thématique fournie similaire est 
    "Titre : 'signature électronique' | Lien : https://www.docaposte.com/solutions/signature-electronique", 
    et que le texte contient "Pour sécuriser vos documents, une signature électronique est essentielle...", 
    vous pourrez mettre dans le résumé un passage comme suit : "Pour sécuriser vos documents, une signature 
    électronique (https://www.docaposte.com/solutions/signature-electronique) est essentielle...". Il faut 
    nécessairement que le lien inséré entre parenthèses soit après un mot ou groupe de mots présent dans le titre de 
    la thématique similaire. 
    13. Chaque lien vers les thématiques similaires abordées dans les pages de Docaposte ne 
    doit être inséré qu'une seule fois dans le texte et l'insertion devra être naturelle, subtile, en évitant les 
    expressions du type : "Pour en savoir plus, consulter cette page (lien) ...". Il faut nécessairement insérer tous 
    les liens dans le texte.
    
    ### Sortie : 
    """
    return prompt
