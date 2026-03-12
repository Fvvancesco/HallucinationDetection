# config/prompts.py
PROMPT_REGISTRY = {
    "base_v1": {
        "system": "You are a helpful assistant. Answer truthfully with only one word: TRUE or FALSE.",
        "user": "Is the following statement true: {fact}?"
    },
    "strict_evaluator": {
        "system": "You are a strict logic evaluator. Respond only with 'TRUE' or 'FALSE'. No yapping.",
        "user": "Fact: {fact}\nEvaluation:"
    },
    "cot_simulated": {
        "system": "You are an expert fact-checker.",
        "user": "Think step by step, then tell me if this is true or false: {fact}"
    }
}