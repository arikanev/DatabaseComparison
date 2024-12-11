import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Step 2: Initialize Pinecone client
api_key = "3634f9bb-5fa4-48c2-8dae-081f51654f4c"  # Replace with your API key
pc = Pinecone(api_key=api_key)

# Step 3: Generate sample data and embeddings
data = [
    # Apple Inc. examples
    {"id": "inc1", "text": "The tech company Apple is known for its innovative products like the iPhone.", "label": "Apple Inc"},
    {"id": "inc2", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.", "label": "Apple Inc"},
    {"id": "inc3", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.", "label": "Apple Inc"},
    {"id": "inc4", "text": "Apple's stock value has increased significantly due to its high sales.", "label": "Apple Inc"},
    {"id": "inc5", "text": "The latest Apple MacBook is incredibly fast and has a great display.", "label": "Apple Inc"},
    {"id": "inc6", "text": "Apple's latest iPhone model has set new records in global sales.", "label": "Apple Inc"},
    {"id": "inc7", "text": "The new Apple Watch includes advanced health-tracking features.", "label": "Apple Inc"},
    {"id": "inc8", "text": "Apple's software ecosystem, including iOS and macOS, is known for its security and ease of use.", "label": "Apple Inc"},
    {"id": "inc9", "text": "Apple Inc. has opened a new campus in Silicon Valley to expand its operations.", "label": "Apple Inc"},
    {"id": "inc10", "text": "Apple's app store generates billions in revenue from app sales and subscriptions.", "label": "Apple Inc"},
    {"id": "inc11", "text": "The Apple Pencil is a popular accessory for digital artists using iPads.", "label": "Apple Inc"},
    {"id": "inc12", "text": "Apple's AirPods are among the most popular wireless earphones worldwide.", "label": "Apple Inc"},
    {"id": "inc13", "text": "Apple Inc. has announced its goal to be carbon neutral by 2030.", "label": "Apple Inc"},
    {"id": "inc14", "text": "The Apple TV+ streaming service features exclusive movies and shows.", "label": "Apple Inc"},
    {"id": "inc15", "text": "Apple's Siri is one of the most widely used virtual assistants on mobile devices.", "label": "Apple Inc"},
    # Apple Fruit examples
    {"id": "fruit1", "text": "Apple is a popular fruit known for its sweetness and crisp texture.", "label": "Apple Fruit"},
    {"id": "fruit2", "text": "Many people enjoy eating apples as a healthy snack.", "label": "Apple Fruit"},
    {"id": "fruit3", "text": "An apple a day keeps the doctor away.", "label": "Apple Fruit"},
    {"id": "fruit4", "text": "Apple pie is a classic dessert loved by many.", "label": "Apple Fruit"},
    {"id": "fruit5", "text": "The apple tree produces delicious fruit enjoyed worldwide.", "label": "Apple Fruit"},
    {"id": "fruit6", "text": "Apples come in various types, including Fuji, Granny Smith, and Gala.", "label": "Apple Fruit"},
    {"id": "fruit7", "text": "The nutritional value of an apple makes it a great snack option.", "label": "Apple Fruit"},
    {"id": "fruit8", "text": "Apples can be used to make cider, a popular fermented beverage.", "label": "Apple Fruit"},
    {"id": "fruit9", "text": "Apple trees are commonly found in temperate regions around the world.", "label": "Apple Fruit"},
    {"id": "fruit10", "text": "Baking apples into pies is a traditional use in many households.", "label": "Apple Fruit"},
    {"id": "fruit11", "text": "An apple's skin is rich in fiber, which aids in digestion.", "label": "Apple Fruit"},
    {"id": "fruit12", "text": "Many farmers grow apples as part of their orchard crops.", "label": "Apple Fruit"},
    {"id": "fruit13", "text": "Applesauce is often enjoyed as a healthy side dish or snack.", "label": "Apple Fruit"},
    {"id": "fruit14", "text": "Eating an apple provides essential vitamins like vitamin C.", "label": "Apple Fruit"},
    {"id": "fruit15", "text": "Some apple varieties are better suited for cooking, while others are best eaten fresh.", "label": "Apple Fruit"}
]

# Convert text to embeddings
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print("Generated Embeddings:", embeddings)

# Prepare data for classification
embedding_values = np.array([e['values'] for e in embeddings])  # Convert embeddings to NumPy array
labels = np.array([1 if d['label'] == "Apple Inc" else 0 for d in data])  # Label "Apple Inc" as 1, "Apple Fruit" as 0

# Step 4: Train a Logistic Regression classifier on the embeddings
classifier = LogisticRegression()
classifier.fit(embedding_values, labels)
predicted_labels = classifier.predict(embedding_values)
accuracy = accuracy_score(labels, predicted_labels)
print(f"Training Accuracy of Logistic Regression on 1024 dimensional data: {accuracy * 100:.2f}%")

# Step 5: Reduce the dimensionality to 3D using t-SNE
tsne = TSNE(n_components=3, random_state=0, perplexity=3)
embeddings_3d = tsne.fit_transform(embedding_values)  # Reduce original embeddings to 3D

# Step 6: Visualization of embeddings in 3D with color-coded labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point with color according to its true label
for i, point in enumerate(embeddings_3d):
    color = 'blue' if labels[i] == 1 else 'green'  # Apple Inc in blue, Apple Fruit in green
    ax.scatter(point[0], point[1], point[2], color=color, label=data[i]['label'] if i < 2 else "", s=50)
    ax.text(point[0], point[1], point[2], data[i]['id'], color='red', fontsize=8)  # Label each point with its ID

# Set plot labels and title
ax.set_xlabel("TSNE Component 1")
ax.set_ylabel("TSNE Component 2")
ax.set_zlabel("TSNE Component 3")
plt.title("3D Visualization of Text Embeddings")

# Show only one legend entry for each label
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
