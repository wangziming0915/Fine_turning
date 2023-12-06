import pandas as pd
import json
import subprocess

# Step 1: Preparing the Data and Launching the Fine Tuning
n = 2000  # Number of rows to read from the Excel file
xlsx_file = 'Medicine_description.xlsx'

df = pd.read_excel(xlsx_file, sheet_name='Sheet1', header=0, nrows=n)
reasons = df["Reason"].unique()
reasons_dict = {reason: i for i, reason in enumerate(reasons)}

df["Drug_Name"] = "Drug: " + df["Drug_Name"] + "\n" + "Malady:"
df["Reason"] = " " + df["Reason"].apply(lambda x: "" + str(reasons_dict[x]))
df.drop(["Description"], axis=1, inplace=True)
df.rename(columns={"Drug_Name": "prompt", "Reason": "completion"}, inplace=True)

jsonl_data = df.to_json(orient="records", indent=0, lines=True)

with open("drug_malady_data.jsonl", "w") as f:
    f.write(jsonl_data)

# Step 2: Command to Prepare Data
subprocess.run(["openai", "tools", "fine_tunes.prepare_data", "-f", "drug_malady_data.jsonl"], check=True)

# Step 3: Command to Train the Model
openai_api_key = "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"  # Replace with your OpenAI API Key
subprocess.run([
    "openai", "api", "fine_tunes.create",
    "-t", "drug_malady_data_prepared_train.jsonl",
    "-v", "drug_malady_data_prepared_valid.jsonl",
    "--compute_classification_metrics",
    "--classification_n_classes", "3",
    "-m", "ada",
    "--suffix", "drug_malady_data"
], check=True, env={"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": openai_api_key})

# Step 4: Checking Job Progress
# Uncomment and replace <JOB ID> with the actual job ID if needed
subprocess.run(["openai", "api", "fine_tunes.follow", "-i", "<JOB ID>"])

# Step 5: Completion of Fine-Tuning
# After completion, follow the instructions provided in the guide

# Example:
# openai api completions.create -m ada:ft-learninggpt:drug-malady-data-2023-02-21-20-36-07 -p "<YOUR_PROMPT>"
