import json
from pathlib import Path
import spacy
import random
try:
    from transformers import pipeline
    _has_transformers = True
except ImportError:
    _has_transformers = False
try:
    from sentence_transformers import SentenceTransformer, util
    _has_st = True
except ImportError:
    _has_st = False

nlp = spacy.load("en_core_web_sm")

intent_labels = [
    "greeting",
    "elections",
    "leaders",
    "parties",
    "policies",
    "scandals",
    "international",
    "economy",
    "protests",
    "history",
    "media",
    "public_opinion",
    "farewell"
]

def load_responses():
    responses_path = Path(__file__).parent / "responses.json"
    with open(responses_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_qa_dataset():
    qa_path = Path(__file__).parent / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)

responses = load_responses()
qa_dataset = load_qa_dataset() if _has_st else []

if _has_st:
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_questions = [item['question'] for item in qa_dataset]
    qa_embeddings = st_model.encode(qa_questions, convert_to_tensor=True)
else:
    st_model = None
    qa_embeddings = None

# Zero-shot intent classifier
if _has_transformers:
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
else:
    zero_shot_classifier = None


def get_intent(text):
    if zero_shot_classifier:
        result = zero_shot_classifier(text, intent_labels)
        if result and result['scores'][0] > 0.4:
            return result['labels'][0]
        else:
            return None
    doc = nlp(text.lower())
    for label in intent_labels:
        if label in text.lower():
            return label
    return None

# Q&A retrieval: find best answer from dataset
def retrieve_qa_answer(user_input, threshold=0.7):
    if not (_has_st and qa_dataset and st_model):
        return None
    user_emb = st_model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_emb, qa_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score > threshold:
        return qa_dataset[best_idx]['answer']
    return None

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def chatbot():
    print("Politix Chatbot: Hi! I'm your political chatbot. Ask me anything about politics.")
    last_intent = None
    while True:
        user_input = input("You: ")
        if not user_input.strip():
            continue
        # Try Q&A retrieval first
        qa_answer = retrieve_qa_answer(user_input)
        if qa_answer:
            print("Politix Chatbot (Q&A):", qa_answer)
            continue
        intent = get_intent(user_input)
        entities = extract_entities(user_input)
        if intent == "farewell":
            print("Politix Chatbot:", random.choice(responses[intent]))
            break
        elif intent:
            if entities:
                entity_str = ', '.join([f'"{e[0]}" ({e[1]})' for e in entities])
                print(f"Politix Chatbot: {random.choice(responses[intent])} By the way, you mentioned {entity_str}.")
            else:
                print("Politix Chatbot:", random.choice(responses[intent]))
            last_intent = intent
        else:
            if last_intent and last_intent in responses:
                print("Politix Chatbot (context):", random.choice(responses[last_intent]))
            else:
                print("Politix Chatbot:", random.choice(responses["default"]))
        if entities:
            print("[NLP] I detected these entities:", entities)

if __name__ == "__main__":
    chatbot()
