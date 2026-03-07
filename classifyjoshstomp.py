import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

DATA_FOLDER = "data/"
DATA_SESSION = "3-4/josh/"
SESSIONS = [5, 6]
CHANNELS = [2, 3, 4, 6] # in theory these are the only ones that should matter

alltrials = np.empty((0, 2))
for session in SESSIONS:
    filtered_session = np.load(f"{DATA_FOLDER}{DATA_SESSION}filtered-session-{session}.npy", allow_pickle=True)
    # trim and select channels
    for i in range(len(filtered_session)):
        filtered_session[i][1] = filtered_session[i][1][CHANNELS, 62:-100]
    alltrials = np.concatenate((alltrials, filtered_session), axis=0)

labels = np.array([1 if trial[0] == 'stomp right' else 0 for trial in alltrials])
eeg = np.array([trial[1] for trial in alltrials])

print(labels.shape)
print(eeg.shape)

csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

pl = Pipeline([
    ('csp', csp),
    ('lda', lda)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pl, eeg, labels, cv=cv, scoring='accuracy')
print(f"fold scores: {scores}")

# train test split
X_train, X_test, y_train, y_test = train_test_split(eeg, labels, test_size=0.25, random_state=67)
pl.fit(X_train, y_train)
train_score = pl.score(X_train, y_train)
print(f"train score: {train_score}")
test_score = pl.score(X_test, y_test)
print(f"test score: {test_score}")

# save fitted model
joblib.dump(pl, f"models/3-4joshstomp.pkl")