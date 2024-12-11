# Database Performance Comparison: PostgreSQL vs. MongoDB

This project evaluates the performance of PostgreSQL and MongoDB for managing multimodal data from the YouTube-8M dataset. By focusing on query performance and consistency, this project provides insights into database selection for applications in video analytics and recommendation systems.

## Overview

- **Dataset:** YouTube-8M
  - A large-scale dataset with millions of video annotations.
  - Contains metadata (IDs, labels) and binary data (mean RGB and audio embeddings).
  - Stored in TFRecord format, converted to database-compatible format for this study.
- **Databases Evaluated:**
  - PostgreSQL
  - MongoDB

## Project Goals

1. Compare query performance and consistency of PostgreSQL and MongoDB.
2. Provide actionable insights for database selection in applications dealing with multimodal data.
3. Share methodologies, scripts, and findings through this repository.

## Challenges

- **Data Conversion:** Transformed TFRecord files to formats compatible with PostgreSQL and MongoDB.
- **MongoDB Setup:** Hosted a MongoDB cluster on Atlas (AWS) and managed data import using the command line.
- **PostgreSQL Queries:** Optimized SQL queries for performance evaluation.

## Key Performance Metrics

1. **Basic Read Operations**
2. **Filtered Queries (Selective Search)**
3. **Aggregation Performance**
4. **Complex Queries**

### Summary of Findings

- MongoDB excelled in basic read operations.
- PostgreSQL was significantly faster in aggregation and complex queries.
- Filtered query performance was similar, with PostgreSQL having a slight edge.
- MongoDB demonstrated more consistent performance (lower relative standard deviation).

## Tools and Technologies

- **Databases:** PostgreSQL, MongoDB
- **Data Processing:** Python (pandas, matplotlib)
- **Version Control:** Git
- **Dataset:** [YouTube-8M Dataset](https://www.kaggle.com/datasets/youtube/youtube-8m)

## Repository Structure

- `scripts/`: Database setup and query scripts.
- `data/`: Processed dataset files.
- `results/`: Performance analysis and comparison results.
- `docs/`: Documentation and final report.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/database-performance-comparison.git
