import os

GEN_OUTDIR = f"/private/home/{os.environ.get('USER')}/uxr_study_data/m2m100_generations"

LANGUAGES = ["az"]

def main():
    paragraph_start_indices = construct_paragraph_start_indices()

    for lang in LANGUAGES:
        FILEFORMAT = f"{GEN_OUTDIR}/en_{lang}_beam4.out"
        with open(FILEFORMAT) as f:
            data = f.readlines()
        translations = {}
        for line in data:
            if line[0:2] == "H-":
                translation = line.strip().split("\t")[2]
                line_index = int(line.strip().split("\t")[0].split("-")[1])
                translations[line_index] = translation
        assert(len(translations) == 122)

        paragraphs = []
        current_paragraph_parts = []
        for sentence_index in range(len(translations)):
            next_paragraph_number = len(paragraphs) + 1
            if next_paragraph_number < len(paragraph_start_indices) and \
               paragraph_start_indices[next_paragraph_number] == sentence_index and \
               len(current_paragraph_parts) > 0:
                paragraphs.append(" ".join(current_paragraph_parts))
                current_paragraph_parts = []
            current_paragraph_parts.append(translations[sentence_index])

        # Make sure to write the last paragraph being constructed.
        if len(current_paragraph_parts) > 0:
            paragraphs.append(" ".join(current_paragraph_parts))
            current_paragraph_parts = []

        with open(f"{lang}_paragraphs.txt", "w") as o:
            for line in paragraphs:
                o.write(f"{line}\n")
        # reorder the translations
        # reconstruct

def construct_paragraph_start_indices():
    with open("english_sentences.txt") as f:
        sentence_data = [line.strip() for line in f.readlines()]

    with open("english_paragraphs.txt") as f:
        paragraph_data = [line.strip() for line in f.readlines()]

    start_indices = []

    next_paragraph = 0
    for i, sentence in enumerate(sentence_data):
        if next_paragraph < len(paragraph_data) and paragraph_data[next_paragraph].startswith(sentence):
            start_indices.append(i)
            next_paragraph += 1

    return start_indices

if __name__ == "__main__":
    main()