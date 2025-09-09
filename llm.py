from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,

)
tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

messages = [
    {"role": "user", "content": "Create a funny joke about cats."},
]

output = generator(messages)
print(output[0]['generated_text'])
