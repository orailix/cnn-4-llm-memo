# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.


def parse_single_title(title: str, dupl_thd=None) -> str:

    if title == "other":
        return "Non-Memo"

    if title == "recite" and dupl_thd is not None:
        if dupl_thd == 1000:
            dupl_thd = "1k"

        return f"Recite[{dupl_thd}]"

    if title == "recollect" and dupl_thd is not None:
        if dupl_thd == 1000:
            dupl_thd = "1k"

        return f"Recollect[{dupl_thd}]"

    return title.capitalize()


def get_titles_no_others(tax_name, dupl_thd=None):

    if tax_name[: len("merge_1_")] == "merge_1_":
        categories = [
            parse_single_title(elt, dupl_thd)
            for elt in tax_name[len("merge_1_") :].split("_")
        ]
        return [categories[0], f"{categories[1]}-or\n-{categories[2]}", categories[3]]

    if tax_name[: len("merge_2_")] == "merge_2_":
        categories = [
            parse_single_title(elt, dupl_thd)
            for elt in tax_name[len("merge_2_") :].split("_")
        ]
        return [categories[0], categories[1], f"{categories[2]}-or\n-{categories[3]}"]

    return [parse_single_title(elt, dupl_thd) for elt in tax_name.split("_")]


def get_titles(tax_name, dupl_thd=None, remove_linebreak: bool = False):

    result = get_titles_no_others(tax_name, dupl_thd)

    if remove_linebreak:
        result = ["".join(item.split("\n")) for item in result]

    return result[:-1] + ["Others"]
