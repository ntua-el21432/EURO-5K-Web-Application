import re
import spacy
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens import Doc

# Set of common legal abbreviations to prevent splitting
LEGAL_ABBR = {"Art.", "Reg.", "Dir.", "No.", "Sec.", "Annex.", "Art", "Reg", "Dir", "Sec", "Annex"}

@Language.component("merge_multilevel_numbers")
def merge_multilevel_numbers(doc: Doc) -> Doc:
    """Merges complex numbering tokens (e.g. 1.2, 6.2.2)."""
    spans_to_merge = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if not token.text.isdigit():
            i += 1
            continue

        start = i
        i += 1
        while i < len(doc) and doc[i].text in ". ":
            i += 1
        while i < len(doc) and doc[i].text.isdigit():
            i += 1
            while i < len(doc) and doc[i].text in ". ":
                i += 1

        candidate_text = doc[start:i].text.replace(" ", "")
        if re.fullmatch(r"\d+(\.\d+\.?){1,5}", candidate_text): 
            spans_to_merge.append(doc[start:i])

    if spans_to_merge:
        with doc.retokenize() as retokenizer:
            for span in spans_to_merge:
                retokenizer.merge(span)

    # NEW SENTENCE only when the number is very likely a heading
    for token in doc:
        text = token.text
        if re.match(r"^\d+\.\d", text.rstrip(".")):
            if token.i + 1 >= len(doc):
                continue
            next_token = doc[token.i + 1]
            prev_token = doc[token.i - 1] if token.i > 0 else None
            prev_text = prev_token.text.lower() if prev_token else ""

            if prev_text in {"point", "points", "article", "articles", "annex", "annexes",
                             "recital", "recitals", "paragraph", "paragraphs", "section",
                             "sections", "item", "items", "," , "and", "or", "to", "("}:
                continue

            if next_token.text and next_token.text[0].isupper():
                next_token.is_sent_start = True
    return doc

@Language.component("block_numbering")
def block_numbering(doc: Doc) -> Doc:
    """
    Prevents splitting after simple list numbers like '3.' 
    BUT allows splitting if the number is likely part of the previous text.
    """
    for i in range(len(doc) - 2):
        # Look for pattern: [Number] [.] [Any Token]
        if doc[i].like_num and doc[i+1].text == ".":
            
            # --- LOGIC CHANGE HERE ---
            # If the number is preceded by a non-punctuation word (e.g., "Article 5."),
            # assume the dot is a Full Stop and ALLOW the split.
            if i > 0:
                prev = doc[i-1]
                # If previous token is a word (not space/punct), it's likely end of sentence.
                # Example: "...and 63. It..." -> prev is 'and'.
                if not prev.is_space and not prev.is_punct:
                    continue # Skip blocking -> Default behavior is to split
            
            # Otherwise (Start of line, or after newline/punct), treat as List Item
            # Example: "\n3. The..." -> Block split
            doc[i+2].is_sent_start = False
            
    return doc

@Language.component("legal_fix")
def legal_fix(doc: Doc) -> Doc:
    """Prevents splitting after legal abbreviations."""
    for i, token in enumerate(doc[:-1]):
        if token.text in LEGAL_ABBR:
            doc[i+1].is_sent_start = False
        if token.text.lower() in {"article", "annex", "section"} and doc[i+1].like_num:
            if i+2 < len(doc):
                doc[i+2].is_sent_start = False
    return doc

def build_legal_splitter():
    print("Building legal sentence splitter NLP pipeline...")
    nlp = spacy.blank("en")
    for abbr in LEGAL_ABBR:
        nlp.tokenizer.add_special_case(abbr, [{"ORTH": abbr}])
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("merge_multilevel_numbers")   
    nlp.add_pipe("block_numbering", after="sentencizer")
    nlp.add_pipe("legal_fix", after="sentencizer")
    return nlp

_legal_nlp = build_legal_splitter()

def split_sentences(text):
    if not text or not text.strip():
        return []
    try:
        doc = _legal_nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        print(f"Error in split_sentences: {e}")
        return text.split('\n')