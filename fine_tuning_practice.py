import json
import subprocess
import openai

# Step 1: Activate Python Environment and Set API Key
openai_api_key = "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"
subprocess.run(["export", f"OPENAI_API_KEY={openai_api_key}"], shell=True)

# Step 2: Create JSONL File
data = [
    {"prompt": "When do I have to start the heater?", "completion": "Every day in the morning at 7AM. You should stop it at 2PM"},
    {"prompt": "Where is the garage remote control?", "completion": "Next to the yellow door, on the key ring"},
    {"prompt": "Is it necessary to program the scent diffuser every day?", "completion": "The scent diffuser is already programmed, you just need to recharge it when its battery is low"}
]

with open("data.jsonl", "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write('\n')

# Step 3: Analyze and Prepare Data
subprocess.run(["openai", "tools", "fine_tunes.prepare_data", "-f", "data.json"], check=True)

# Step 4: Fine-Tune the Model
subprocess.run(["openai", "api", "fine_tunes.create", "-t", "data_prepared.jsonl", "-m", "curie"], check=True)

# Step 5: Resume Fine-Tuning (if needed)
# Uncomment and replace <YOUR_FINE_TUNE_JOB_ID> with the actual job ID if needed
subprocess.run(["openai", "api", "fine_tunes.follow", "-i", ""], check=True)

# Step 6: List Fine-Tuned Models
subprocess.run(["openai", "api", "fine_tunes.list"])

# Step 7: Use the Fine-Tuned Model

subprocess.run(["openai", "api", "completions.create", "-m", "curie:ft-learninggpt-2023-02-18-08-38-08", "-p", "<YOUR_PROMPT>"])

# Step 8: Use Fine-Tuned Model in Python
# Uncomment and replace <FINE_TUNED_MODEL> and <YOUR_PROMPT> with the actual model and prompt
openai.Completion.create(
     model="<FINE_TUNED_MODEL>",
     prompt="<YOUR_PROMPT>"
 )

# Step 9: Use Fine-Tuned Model with cURL
# Uncomment and replace <FINE_TUNED_MODEL> and <YOUR_PROMPT> with the actual model and prompt
subprocess.run(['curl', 'https://api.openai.com/v1/completions', '-H', f'Authorization: Bearer {openai_api_key}', '-H', 'Content-Type: application/json', '-d', f'{{"prompt": "<YOUR_PROMPT>", "model": "gpt-3.5-turbo"}}'])

# Step 10: Analyze Fine-Tuned Model
# Uncomment and replace <YOUR_FINE_TUNE_JOB_ID> with the actual job ID
subprocess.run(["openai", "api", "fine_tunes.results", "-i", ""])

# Step 11: Add Suffix to Fine-Tuned Model Name
# Uncomment and replace <engine> with the engine you used for fine-tuning
subprocess.run(['openai', 'api', 'fine_tunes.create', '-t', 'data.jsonl', '-m', 'gpt-3.5-turbo', '--suffix', 'my_model_name'])

# Step 12: Delete Fine-Tuned Model
# Uncomment and replace <FINE_TUNED_MODEL> with the actual model
subprocess.run(['openai', 'api', 'models.delete', '-i', 'gpt-3.5-turbo'])
