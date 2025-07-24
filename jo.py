import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === GPT-4-turbo (via OpenAI API) ===
openai.api_key = "your-openai-api-key"

def ask_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# === Open-source model (e.g. Mistral or LLaMA) ===
def ask_open_source_model(prompt, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    response = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].strip()

# === Main comparison ===
prompt = "Explain photosynthesis to a high school student."

print("=== GPT-4-turbo response ===")
print(ask_gpt4(prompt))

print("\n=== Open-source model response ===")
print(ask_open_source_model(prompt))

