# BrainCrafterInterface
An EEG-based BCI that classifies motor signals generated from movement and uses them to press keys, allowing movement-based control for different games (we tested it on Minecraft, but theoretically it should work for other games too). 

This was made for the final project for COGS 189 (WI26). Our write-up is is available [here](https://docs.google.com/document/d/1PUnr1EmVb9j7FB7ZqH1a4syJxg2aweJLtR3ka0RZq6A/edit?usp=sharing)
(only available to those with UCSD SSO credentials).

### Demo
[Click here for a demo video!](https://drive.google.com/file/d/1sQcX_kibuCumDX5bbVyPhzxyUVgbsHN7/view?usp=sharing) (Only available to those with UCSD SSO credentials)

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

## 🧠 How it works
1. A participant is fitted with an EEG cap and the following EEG channels are gelled, prepped, and connected to a cyton (in order): F3, F4, C3, Cz, C4, P3, Pz, P4. The Fz electrode is used as the reference and the GND electrode is used as the ground.
2. Data is collected from participants using the `collect_data.py` script; by default this will collect one session consisting of a 20-second baseline followed by 20 trials of 4 seconds each sampled at 250 Hz.
    * These parameters can be changed at the top of the script.
    * See the additional notes below for important details on data collection.
    * The following will be saved as numpy arrays: `aux-continuous`, `baseline`, `eeg-continguous`, `filtered-baseline`, `filtered-session`, and `raw-session`.
    * In the filtered arrays, third-order Butterworth bandpass filter is applied to the data (8-30 Hz).
3. Given that each participant differs in their brain signals, and that each motor action produces different signals, we train a specific (but similar for our use case) classifier for each participant and motor action. We use the `classify_*.py` scripts for this.
4. After evaluating the performance of our classifiers, we can use the `livetest.py` script to load the model and classify data in real-time and press keys based on the predicted motor action.


## 📝 Additional notes
Please run any scripts from the top level directory (i.e. `BrainCrafterInterface` unless you cloned the repo somewhere else). 
- In particular, the terminal command will look like `python scripts/script.py`.
- If you are in the `scripts/` directory or using VSCode, you can adjust the relative paths (i.e. change `data/` to `../data/` and `models/` to `..models/`).

### Data collection
Example code from class is in the `OpenVEP/` directory.

BEFORE running the code, make sure you update the following variables in `collect_data.py`:
- `DATA_DIR`: keep of form `data/{date of lab session}`
- `SES_NUMBER`: number of session of running code/collecting data 

> [!CAUTION]
> If you run the script multiple times without changing `SES_NUMBER` it will overwrite your previous data!

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

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Also, see the [NOTICE](NOTICE) file for details on third-party software used in this project.