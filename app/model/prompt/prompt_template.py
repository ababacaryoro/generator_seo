import os


def get_prompt_template(name="prompt_rag.txt"):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the absolute path to the prompt_sumup.txt file
    file_path = os.path.join(dir_path, name)

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
