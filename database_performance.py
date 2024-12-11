import time
import statistics
import psycopg2
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

class DBPerformanceTester:
    def __init__(self):
        # MongoDB Atlas connection
        self.mongo_client = MongoClient("mongodb+srv://arielkanevsky97:U1tqY4u3jtIppFjm@cluster0.ez6ey.mongodb.net/")
        self.mongo_db = self.mongo_client['databases']
        self.mongo_collection = self.mongo_db['youtube8m']
        
        # PostgreSQL connection
        self.pg_conn = psycopg2.connect(
            dbname="postgres",
            user="arikanevsky",
            password="Aribenjamin1997!",
            host="localhost",
            port="5432"
        )
        self.pg_cursor = self.pg_conn.cursor()
        
        # Results storage
        self.results = []

    def measure_query_time(self, func, iterations=5):
        times = []
        for _ in range(iterations):
            try:
                start = time.time()
                func()
                self.pg_conn.commit()  # Commit after each query
                end = time.time()
                times.append(end - start)
            except Exception as e:
                print(f"Query error: {str(e)}")
                self.pg_conn.rollback()  # Rollback on error
                raise
        
        return {
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0
        }

    def test_basic_read(self, limit=1000):
        print(f"\nTesting Basic Read (limit {limit})...")
        
        # PostgreSQL test
        pg_query = f"SELECT * FROM youtube8m LIMIT {limit}"
        pg_times = self.measure_query_time(
            lambda: self.pg_cursor.execute(pg_query)
        )
        
        # MongoDB test
        mongo_times = self.measure_query_time(
            lambda: list(self.mongo_collection.find().limit(limit))
        )
        
        self.results.append({
            'test_name': 'basic_read',
            'parameters': {'limit': limit},
            'postgresql': pg_times,
            'mongodb': mongo_times
        })

    def test_filtered_search(self, label_value=180):
        print(f"\nTesting Filtered Search (label={label_value})...")
        
        # PostgreSQL test - modified for text search
        pg_query = f"SELECT * FROM youtube8m WHERE labels LIKE '%{label_value}%'"
        pg_times = self.measure_query_time(
            lambda: self.pg_cursor.execute(pg_query)
        )
        
        # MongoDB test remains the same
        mongo_times = self.measure_query_time(
            lambda: list(self.mongo_collection.find({"labels": label_value}))
        )
        
        self.results.append({
            'test_name': 'filtered_search',
            'parameters': {'label_value': label_value},
            'postgresql': pg_times,
            'mongodb': mongo_times
        })

    def test_aggregation(self, limit=10):
        print(f"\nTesting Aggregation (top {limit} labels)...")
        
        # PostgreSQL test - modified for text handling
        pg_query = f"""
        SELECT labels, COUNT(*) 
        FROM youtube8m 
        GROUP BY labels 
        ORDER BY COUNT(*) DESC 
        LIMIT {limit}
        """
        pg_times = self.measure_query_time(
            lambda: self.pg_cursor.execute(pg_query)
        )
        
        # MongoDB test remains the same
        mongo_pipeline = [
            {"$group": {"_id": "$labels", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        mongo_times = self.measure_query_time(
            lambda: list(self.mongo_collection.aggregate(mongo_pipeline))
        )
        
        self.results.append({
            'test_name': 'aggregation',
            'parameters': {'limit': limit},
            'postgresql': pg_times,
            'mongodb': mongo_times
        })

    def test_complex_query(self):
        print("\nTesting Complex Query...")
        
        # PostgreSQL test - modified for text column
        pg_query = """
        WITH LabelCounts AS (
            SELECT labels, COUNT(*) as cnt
            FROM youtube8m
            GROUP BY labels
        )
        SELECT labels, cnt
        FROM LabelCounts
        WHERE cnt > (SELECT AVG(cnt) FROM LabelCounts)
        ORDER BY cnt DESC;
        """
        
        pg_times = self.measure_query_time(
            lambda: self.pg_cursor.execute(pg_query)
        )
        
        # MongoDB test
        mongo_pipeline = [
            {"$group": {"_id": "$labels", "count": {"$sum": 1}}},
            {
                "$group": {
                    "_id": None,
                    "labels": {"$push": {"label": "$_id", "count": "$count"}},
                    "avg": {"$avg": "$count"}
                }
            },
            {"$unwind": "$labels"},
            {"$match": {"labels.count": {"$gt": "$avg"}}},
            {"$project": {"label": "$labels.label", "count": "$labels.count"}},
            {"$sort": {"count": -1}}
        ]
        
        mongo_times = self.measure_query_time(
            lambda: list(self.mongo_collection.aggregate(mongo_pipeline))
        )
        
        self.results.append({
            'test_name': 'complex_query',
            'parameters': {},
            'postgresql': pg_times,
            'mongodb': mongo_times
        })

    def create_indexes(self):
        print("\nCreating indexes...")
        try:
            # First check the column types
            self.pg_cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'youtube8m';
            """)
            columns = self.pg_cursor.fetchall()
            print("Column types:", columns)
            
            # For text column, use btree index
            pg_index = """
            CREATE INDEX IF NOT EXISTS idx_labels 
            ON youtube8m USING btree (labels);
            """
            
            self.pg_cursor.execute(pg_index)
            self.pg_conn.commit()
            
            # MongoDB index
            self.mongo_collection.create_index([("labels", 1)])
            
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            # Rollback the failed transaction
            self.pg_conn.rollback()


    def generate_report(self):
        print("\nGenerating performance report...")
        
        # Convert results to DataFrame for easy analysis
        rows = []
        for result in self.results:
            row = {
                'Test': result['test_name'],
                'Parameters': str(result['parameters']),
                'PostgreSQL Avg (s)': result['postgresql']['avg'],
                'PostgreSQL Min (s)': result['postgresql']['min'],
                'PostgreSQL Max (s)': result['postgresql']['max'],
                'PostgreSQL Std (s)': result['postgresql']['std'],
                'MongoDB Avg (s)': result['mongodb']['avg'],
                'MongoDB Min (s)': result['mongodb']['min'],
                'MongoDB Max (s)': result['mongodb']['max'],
                'MongoDB Std (s)': result['mongodb']['std']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'db_performance_results_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
        # Print summary
        print("\nPerformance Summary:")
        print(df.to_string())

    def run_all_tests(self):
        try:
            # Create indexes first
            self.create_indexes()
            
            # Run all tests
            self.test_basic_read(limit=1000)
            self.test_filtered_search(label_value=180)
            self.test_aggregation(limit=10)
            self.test_complex_query()
            
            # Generate report
            self.generate_report()
            
        finally:
            # Clean up connections
            self.pg_cursor.close()
            self.pg_conn.close()
            self.mongo_client.close()

if __name__ == "__main__":
    # Run tests
    tester = DBPerformanceTester()
    tester.run_all_tests()
