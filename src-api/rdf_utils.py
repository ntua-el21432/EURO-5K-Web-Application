import re
import uuid
from datetime import datetime
import spacy

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Warning: 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# RRMV namespace prefixes
PREFIXES = """@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix eli: <http://data.europa.eu/eli/ontology#> .
@prefix rrmv: <http://data.europa.eu/2qy/rrmv#> .
@prefix example: <https://reporting-obligations.eu/example#> .

"""

# Reporting verbs for dependency analysis
REPORTING_VERBS = [
    "report", "notify", "inform", "submit", "provide", "communicate",
    "transmit", "send", "forward", "deliver"
]

def extract_agents_spacy(text):
    """
    Extract reporting entity and recipients using spaCy dependency parsing
    Returns: (reporter, recipients_list)
    """
    if not nlp:
        return None, []

    doc = nlp(text)
    
    reporter = None
    recipients = []
    
    # Find reporting verb
    for token in doc:
        if token.lemma_.lower() in REPORTING_VERBS:
            # Find subject (reporter) - 'nsubj'
            for child in token.children:
                if child.dep_ == 'nsubj':
                    # Get full noun phrase
                    reporter = ' '.join([t.text for t in child.subtree])
                    break
            
            # Find recipients via prepositional phrases (to X)
            for child in token.children:
                if child.dep_ == 'prep' and child.text.lower() in ['to', 'with']:
                    # Get prep objects
                    for obj in child.children:
                        if obj.dep_ == 'pobj':
                            # Get full noun phrase
                            recipient = ' '.join([t.text for t in obj.subtree])
                            recipients.append(recipient)
                
                # Also check for direct objects (dobj) or dative
                elif child.dep_ in ['dobj', 'dative']:
                    recipient = ' '.join([t.text for t in child.subtree])
                    if any(auth in recipient.lower() for auth in 
                           ['commission', 'authority', 'esma', 'eba', 'member state']):
                        recipients.append(recipient)
        
            if reporter or recipients:
                break
    
    return reporter, recipients

def normalize_entity_name(text):
    """Convert entity text to clean ID (e.g. 'The Member State' -> 'MemberState')"""
    if not text:
        return None
    text = text.lower()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    return ''.join(word.capitalize() for word in words if word)

def text_to_slug(text, max_len=50):
    """Convert text to URL-safe slug"""
    words = text.split()[:6]
    slug = '_'.join(words)
    slug = re.sub(r'[^a-zA-Z0-9_]', '', slug)
    return slug[:max_len]

def convert_to_rrmv(obligation_text, doc_id, index):
    """Convert single obligation to RRMV RDF using spaCy logic"""
    req_id = f"example:Req{index:03d}"
    action_id = f"example:Action{index:03d}"
    reporter, recipients = extract_agents_spacy(obligation_text)
    
    # Normalize to IDs
    reporter_id = normalize_entity_name(reporter) if reporter else None
    recipient_ids = [normalize_entity_name(r) for r in recipients if r]
    
    # Build RDF
    rdf = f"""
# Requirement {index}: {text_to_slug(obligation_text)}

{req_id} a rrmv:Requirement ;
    rrmv:hasURI "http://data.europa.eu/eli/placeholder/{doc_id}"^^xsd:anyURI ;
    dcterms:isPartOf <http://data.europa.eu/eli/placeholder/{doc_id}> ;
    rrmv:produces {action_id} .

{action_id} a rrmv:Action"""
    
    roles = []
    
    if reporter_id:
        roles.append(f"""[
        a rrmv:AgentRole ;
        rrmv:forAgent example:{reporter_id} ;
        rrmv:withRole rrmv:addresser
    ]""")
    
    for r_id in recipient_ids:
        roles.append(f"""[
        a rrmv:AgentRole ;
        rrmv:forAgent example:{r_id} ;
        rrmv:withRole rrmv:addressee
    ]""")
    
    if roles:
        rdf += " ;\n    rrmv:hasAgentRole " + " ,\n    ".join(roles)
    
    rdf += " .\n"
    
    if reporter_id:
        rdf += f"\nexample:{reporter_id} a rrmv:Agent .\n"
    for r_id in recipient_ids:
        rdf += f"example:{r_id} a rrmv:Agent .\n"
    
    rdf += f"""
[] a oa:Annotation ;
    oa:hasBody "{obligation_text}"@en ;
    oa:hasTarget {req_id} .
"""
    return rdf

def generate_rrmv_turtle(predictions):
    """
    Main entry point: Receives list of prediction objects from API 
    and returns full Turtle string.
    """
    rdf_output = PREFIXES
    rdf_output += f"# Generated: {datetime.now().isoformat()}\n"
    rdf_output += "# Method: spaCy dependency parsing\n"
    rdf_output += "# RRMV Version: 1.0.0\n\n"
    
    doc_id = f"doc_{str(uuid.uuid4())[:8]}"
    
    count = 0
    for idx, item in enumerate(predictions, 1):
        text = item.get('text', '')
        rdf_output += convert_to_rrmv(text, doc_id, idx)
        rdf_output += "\n"
        count += 1
        
    return rdf_output