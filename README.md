# BrainCrafterInterface
An EEG-based BCI that classifies motor signals generated from movement and uses them to press keys, allowing movement-based control for different games (we tested it on Minecraft, but theoretically it should work for other games too). 

This was made for the final project for COGS 189 (WI26). Our paper is ~~available at~~ still in the works.

## 📁 Project Structure
Our repo has the following setup:
```
data/                   # collected data separated by date and participant
└── ...
models/                 # classifiers
└── ...
scripts/            
├── classify_*.py       # scripts to train models on our data
├── collect_data.py     # experiment code for collecting data in lab
└── livetest.py         # online functionality (the thing that classifies data in real-time)

viz/                    # notebooks for manipulating & visualizing our data
└── ...
```

### Additional notes
Example code from class is in the `OpenVEP/` directory.

BEFORE running the code, make sure you update the following variables in `collect_data.py`:
- `DATA_DIR`: keep of form `data/{date of lab session}`
- `SES_NUMBER`: number of session of running code/collecting data (**if you run the script multiple times without changing this one it will overwrite your previous data!**)

To collect data using our code:
1. Install dependencies from `requirements.txt`.
2. Run `collect_data.py` (this is the "experiment").
3. Data will be saved to `DATA_DIR`.

## 👤 Team
This project was made possible only by the combined efforts of **Professor X's Team 67**:
- Nathan Tosoc
- Derek Li
- Benjamin Wang
- Miles Davis
- Joshua Weistrop