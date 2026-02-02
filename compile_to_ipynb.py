with open("Agent.py", "r") as file:
    agent = file.readlines()

with open("Environment.py", "r") as file:
    env = file.readlines()

with open("Preprocessor.py", "r") as file:
    preprocessor = file.readlines()

with open("main.py") as file:
    main = file.readlines()

ipynb = {}

ipynb["metadata"]={
  "language_info": {"name": "python"}
  }

ipynb["nbformat"] = 4

ipynb["nbformat_minor"] = 5

ipynb["cells"] = []

for entry in [agent, env, preprocessor, main]:
    ipynb["cells"].append(
        {
        "cell_type": "code",
        "execution_count": None,
        "id": "5a5b11fa",
        "metadata": {"vscode": {"languageId": "plaintext"}},
        "outputs": [],
        "source": entry
        }
    )

import json

with open("compiled_notebook.ipynb", "w") as file:
    json.dump(ipynb, file)
    