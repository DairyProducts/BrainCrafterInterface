# BrainCrafterInterface

Example code from class is in the `OpenVEP/` directory.

BEFORE running the code, make sure you update the following variables in `collect_data.py`:
- `DATA_DIR`: keep of form `data/{date of lab session}`
- `SES_NUMBER`: number of session of running code/collecting data (**if you run the script multiple times without changing this one it will overwrite your previous data!**)

To collect data using our code:
1. Install dependencies from `requirements.txt`
2. Run `collect_data.py` (this is the "experiment")
3. Data will be saved to `DATA_DIR`
