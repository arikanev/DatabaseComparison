import tensorflow as tf
import sqlite3
import os

# Connect to SQLite database
conn = sqlite3.connect('youtube8m.db')
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS videos
                  (id TEXT PRIMARY KEY, labels TEXT, mean_rgb BLOB, mean_audio BLOB)''')

# Function to parse a single example
def parse_example(example):
    features = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32),
        'mean_audio': tf.io.FixedLenFeature([128], tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example, features)
    
    return {
        'id': parsed_features['id'].numpy().decode('utf-8'),
        'labels': ','.join(map(str, parsed_features['labels'].values.numpy())),
        'mean_rgb': parsed_features['mean_rgb'].numpy().tobytes(),
        'mean_audio': parsed_features['mean_audio'].numpy().tobytes()
    }

# Process each TFRecord file
for filename in os.listdir('.'):
    if filename.endswith('.tfrecord'):
        dataset = tf.data.TFRecordDataset(filename)
        
        for raw_record in dataset:
            record = parse_example(raw_record)
            
            cursor.execute('''INSERT OR REPLACE INTO videos (id, labels, mean_rgb, mean_audio)
                              VALUES (?, ?, ?, ?)''',
                           (record['id'], record['labels'], record['mean_rgb'], record['mean_audio']))

# Commit changes and close connection
conn.commit()
conn.close()

print("Data has been loaded into SQLite database.")
