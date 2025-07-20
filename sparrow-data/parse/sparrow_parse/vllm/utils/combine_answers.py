import openai
import os
import json

def combine_answers(key, answers):
    if len(answers) == 0:
        return None
    elif len(answers) == 1:
        return answers[0]
    
    # now use openai to combine the answers
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"We're extracting structured information from a PDF, and have extracted multiple answers from different pages for the field {key}. Combine the following answers for {key} into a single, final answer: {answers}"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "summary",
                "schema": {
                    "type": "object",
                    "properties": {key: {"type": "string"}},
                    "required": [key],
                    "additionalProperties": False
                }
            }
        }
    )

    final_answer = response.choices[0].message.content
    return json.loads(final_answer)


def combine_json_results(json_results):
    combined_result = {}

    final_results = []
    all_keys = set()
    for result in json_results:
        result = json.loads(result)
        final_results.append(result)
        for key in result.keys():
            all_keys.add(key)


    for key in all_keys:
        answers = []
        for result in final_results:
            if key in result and result[key] is not None:
                answers.append(result[key])
        
        # now combine the answers
        combined_result[key] = combine_answers(key, answers)
    
    return combined_result