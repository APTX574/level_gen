import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = 'your-api-key-here'


llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    max_tokens=1500  
)

extraction_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""You are a linguistics expert analyzing educational content. Carefully extract all key technical terms from the following text that require complexity adjustment for different learner levels. Consider:

    1. Specialized vocabulary beyond daily usage
    2. Abstract conceptual terminology
    3. Domain-specific jargon
    4. Terms with complexity variations across cognitive levels
    Return a JSON list without commentary:
    {{
      "terms": ["term1", "term2", ...]
    }}
    Text: {text}
    """
)

extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt_template)

mapping_prompt_template = PromptTemplate(
    input_variables=["terms"],
    template="""You are tasked with categorizing the following technical terms based on their complexity and domain specificity. Organize these terms into a hierarchical structure that reflects their usage in educational contexts, from basic to advanced levels. Return a JSON object with categories as keys and lists of terms as values:
    Terms: {terms}
    Hierarchical Mapping:
    {{
      "Basic": ["term1", "term2", ...],
      "Intermediate": ["term3", "term4", ...],
      "Advanced": ["term5", "term6", ...]
    }}
    """
)

mapping_chain = LLMChain(llm=llm, prompt=mapping_prompt_template)

rewrite_prompt_template = PromptTemplate(
    input_variables=["text", "cognitive_level", "bnf_rules"],
    template="""As a linguistic editor, rewrite the following text strictly adhering to these BNF constraints for {cognitive_level}:  
    {bnf_rules}  
    Key requirements:  
    1. Sentence structure must validate against BNF  
    2. Lexical complexity matches {cognitive_level} terminology  
    3. Preserve original semantic content  

    Input: {text}  
    Output (JSON):  
    {{
      "original": "{text}",
      "restructured": "...", 
      "validation": {{"pass": bool, "issues": []}}
    }}
    """
)

rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt_template)

consistency_prompt_template = PromptTemplate(
    input_variables=["basic", "intermediate", "advanced"],
    template="""
    {{
      "basic": "{basic}",
      "intermediate": "{intermediate}", 
      "advanced": "{advanced}"
    }}
    Instruction:
    "Detect factual conflicts across three cognitive-level sentences. Revise only conflicting parts using strikethrough→correction while preserving original complexity:
    Cross-check scientific accuracy
    Modify contradictions only
    Maintain sentence structure
    Output:
    {{
      "revisions": {{
        "basic": "[revised]",
        "intermediate": "[revised]",
        "advanced": "[revised]"
      }},
    }}
    """
)

consistency_chain = LLMChain(llm=llm, prompt=consistency_prompt_template)

bnf_rules = {
    "Basic": """
    <S> ::= <SimpleNounPhrase> <PresentTenseVerb> <Object>  
    <SimpleNounPhrase> ::= [Determiner] [Adjective] Noun  
    <Object> ::= Noun | "that" <SimpleClause>  
    <SimpleClause> ::= <SimpleNounPhrase> Verb
    """,
    "Intermediate": """
    <S> ::= <ComplexNounPhrase> <VerbPhrase> [Conjunction <S>]  
    <VerbPhrase> ::= [Modal] [Adverb] Verb [PrepositionalPhrase]  
    <ComplexNounPhrase> ::= [Determiner] [Adjective+] Noun [RelativeClause]
    """,
    "Advanced": """
    <S> ::= <Nominalization> | <PassiveVoice> | <Conditional>  
    <PassiveVoice> ::= <NounPhrase> "is" VerbPastParticiple [PrepositionalPhrase]  
    <Conditional> ::= "If" <S> "," ("then" <S> | <ModalVerb> <S>)  
    <Nominalization> ::= <GerundPhrase> Verb <ComplexNounPhrase>    
    <RelativeClause> ::= "that" <VerbPhrase> | "which" <VerbPhrase>
    """
}

def process_answer(answer):

    result = extraction_chain.run({"text": answer})

    extracted_terms = json.loads(result)
    return extracted_terms

def map_terms(terms):

    terms_str = json.dumps(terms)

    result = mapping_chain.run({"terms": terms_str})

    mapped_terms = json.loads(result)
    return mapped_terms

def replace_terms(text, mapping, level):

    for term in mapping[level]:

        replacement = f"{term} ({level})"
        text = text.replace(term, replacement)
    return text

def rewrite_answer(answer, level):

    bnf_rule = bnf_rules[level]
    result = rewrite_chain.run({
        "text": answer,
        "cognitive_level": level,
        "bnf_rules": bnf_rule
    })
    return json.loads(result)

def check_consistency(basic, intermediate, advanced):

    result = consistency_chain.run({
        "basic": basic,
        "intermediate": intermediate,
        "advanced": advanced
    })
    return json.loads(result)

def process_qa_pairs(qa_pairs):
    results = []
    for qa in qa_pairs:
        question = qa['question']
        answer = qa['answer']
        terms = process_answer(answer)
        mapped_terms = map_terms(terms["terms"])
        basic_answer = replace_terms(answer, mapped_terms, "Basic")
        intermediate_answer = replace_terms(answer, mapped_terms, "Intermediate")
        advanced_answer = replace_terms(answer, mapped_terms, "Advanced")
        
        basic_rewrite = rewrite_answer(basic_answer, "Basic")
        intermediate_rewrite = rewrite_answer(intermediate_answer, "Intermediate")
        advanced_rewrite = rewrite_answer(advanced_answer, "Advanced")

        consistency_check = check_consistency(
            basic_rewrite["restructured"],
            intermediate_rewrite["restructured"],
            advanced_rewrite["restructured"]
        )
        
        results.append({
            "question": question,
            "answers": {
                "Basic": basic_rewrite,
                "Intermediate": intermediate_rewrite,
                "Advanced": advanced_rewrite
            },
            "consistency": consistency_check["revisions"]
        })
    return results

qa_pairs = [
    {"question": "What is photosynthesis?", "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."},
]

output = process_qa_pairs(qa_pairs)

print(json.dumps(output, indent=2))
