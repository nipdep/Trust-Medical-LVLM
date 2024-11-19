# Trust-Medical-LVLM

## Folder structure 
```
Trust-Medical-LVLM/
├── data/
│   ├── data/
│   ├── model/
├── src/
│   ├── datasets/
│   ├── evaluators/
│   ├── graders/
│   ├── methods/
│   ├── models/
│   ├── tasks/
│   ├── utils/
├── notebooks/
├── log/
├── scripts/
├── README.md
├── .env
├── .gitignore
└── requirements.txt
```

This is the folder structure for the Trust-Medical-LVLM project.

- `data/`: Contains data-related files and subdirectories.
    - `data/`: Subdirectory for storing raw data files such as CSVs, images, etc.
    - `model/`: Subdirectory for storing pre-trained models, checkpoints, and model outputs.

- `src/`: Contains source code for the project.
    - `configs/`: Config file for datasets and models .
    - `datasets/`: Dataset loading modules.
    - `evaluators/`: Output type post-processing modules, such as YesOrNoEvaluation, ChatBotEvaluation.
    - `graders/`: Generate scores based on post-processed asnwers.
    - `methods/`: Module to pre-process images.
    - `models/`: LVLM model chat interfaces.
    - `tasks/`: Pipelining answer generation, post-processing and generated dataframe saving.
    - `utils/`: Utility functions and helper scripts used throughout the project.

- `notebooks/`: Jupyter notebooks for conducting experiments, visualizations, and analysis.
- `log/`: Directory for storing log files generated during training and evaluation.
- `scripts/`: Scripts for tasks such as data preprocessing, model training, and evaluation.
- `README.md`: The main README file providing an overview and instructions for the project.
- `.env`: File for storing environment variables required for the project.
- `.gitignore`: File specifying which files and directories to ignore in version control.
- `requirements.txt`: File listing all Python dependencies required to run the project.


