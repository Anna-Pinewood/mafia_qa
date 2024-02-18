"""Model service to generate answers."""
from datetime import datetime
from logging import getLogger
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI(
    title=("LLM api."),
    description=("Answers questions based on context."),
    version="0.1.0",
    redoc_url=None,
)

LOGGER = getLogger(__name__)

RUSSIAN_BEGIN_FOR_MODEL = ""


@app.post('/predict')
async def predict(
        request: Request,
):
    """
    Parameters
    ----------
    request.json() is dict, e.g.:
    data = {
        'query': query,
        ...
        'max_tokens': max_new_tokens,
        'temperature': temperature,
    }
    """
    LOGGER.info("\n\nGot request")
    data = await request.json()

    query = data.get("query")
    context = data.get("context")
    system_prompt = data.get("system_prompt")
    context_prompt = data.get("context_prompt")
    max_tokens = data.get("max_tokens")
    temperature = data.get("temperature")

    messages = get_messages(question=query,
                            context=context,
                            system_prompt=system_prompt,
                            context_prompt=context_prompt)

    model_input = answer_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_input += RUSSIAN_BEGIN_FOR_MODEL
    LOGGER.debug("Got model_input:\n%s", model_input)

    if context is not None:
        num_tokens_context = len(
            answer_pipeline.tokenizer(context)['input_ids'])
        LOGGER.debug("Num of tokens in passed context %s:", num_tokens_context)

    num_tokens_total = len(answer_pipeline.tokenizer(model_input)['input_ids'])
    LOGGER.debug("Num of tokens in model_input: %s", str(num_tokens_total))

    outputs = answer_pipeline(
        model_input,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=temperature,
    )

    text = outputs[0]["generated_text"]
    answer = text.split(
        "<|assistant|>")[-1].strip().replace("<|>\n", " ").replace("<|>", " ")
    LOGGER.info("Got response: %s", answer)
    torch.cuda.empty_cache()

    return {"answer": answer}


@app.get('/healthcheck')
async def healthcheck():
    current_time = datetime.now()
    msg = (f"Hey, hey! "
           f"I am LLM service and I've been alive for {current_time - launch_time} now.\n")

    return {"message": msg}


def get_messages(question: str | None = None,
                 context: str | None = None,
                 system_prompt: str | None = None,
                 context_prompt: str | None = None
                 ) -> list[dict[str, str]]:

    history = []
    if system_prompt is not None:
        system_prompt_message = {
            'role': 'system',
            'content': system_prompt
        }
        history = [system_prompt_message] + history

    if context is not None:
        context_prompt = context_prompt + "\n" if context_prompt else ""
        context_text = context_prompt + context

        context_message = {
            'role': 'user',
            'content': context_text
        }
        history.append(context_message)

    if question is not None:
        question_message = {
            "role": 'user',
            "content": question
        }
        history.append(question_message)
    return history


if __name__ == '__main__':
    file_path = Path(__file__)
    default_config_path = file_path.parent / f"{file_path.stem}_config.yaml"
    with open(default_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # pipeline_map = load_pipeline_map(model_configs=model_configs)
    answer_pipeline = pipeline(
        "text-generation",
        model=config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    launch_time = datetime.now()

    uvicorn.run(app, host="0.0.0.0",
                port=config["api_port"], log_level="info", reload=config["reload"], use_colors=True)
