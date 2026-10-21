import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the built-in Iris dataset
print("📦 Loading dataset...")
iris = load_iris()

# Create a clean DataFrame for visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

print("\n--- First 5 Rows of Data ---")
print(df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species_name']].head())

# 2. Split data into Features (X) and Target (y)
X = iris.data
y = iris.target

# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Machine Learning Model
print("\n🤖 Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Make a Live Prediction on New Custom Data
print("🔮 Making a custom prediction...")
# Format: [sepal length, sepal width, petal length, petal width]
new_flower_sample = [[5.1, 3.5, 1.4, 0.2]] 

predicted_class = model.predict(new_flower_sample)[0]
predicted_species = iris.target_names[predicted_class]

print(f"Input features: {new_flower_sample[0]}")
print(f"Predicted Species: ✨ {predicted_species.upper()} ✨")
