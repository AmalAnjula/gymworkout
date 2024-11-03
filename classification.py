import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
 
data = pd.read_csv('joint_angles.csv')

print(data.info())
X = data.drop(columns=['frame', 'workout'])  # Features (excluding 'frame' and 'workout')
y = data['workout']  # Labels
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
 
import joblib
joblib.dump(clf, 'workout_classifier_modeldel.pkl')

train_predictions = clf.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_predictions)}")
