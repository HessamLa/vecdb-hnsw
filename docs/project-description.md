# Open Project: Build a Minimal Viable Vector Database ("VecDB")

## 1. Objective

The goal of this 9-day project is to design and implement a minimal, functional Vector Database. A Vector DB is specialized for storing, indexing, and querying high-dimensional vector embeddings. Your implementation should demonstrate a core understanding of how these databases work under the hood, focusing on efficient similarity search.

Prefer to use C++ as a programming language, but feel free to use other languages.

## 2. Core Requirements

Your VecDB should be a standalone application that provides the following
functionalities through a clear API:
A. Data Management
B. Querying & Search
C. Persistence
    - How data is organized and what indices are supported.
    - Data should not be held only in memory. You must implement a simple persistence layer that saves collections to disk so the database can be reloaded after a restart.

**Bonus points**
- Support saving and searching meta information beside the vector data.
- Consider different access patterns such as OLTP and OLAP.
- Concurrency and Transactions.

## 3. The Challenge: Scalability

A naive implementation would use a brute-force linear scan (comparing the query vector to every vector in the collection), which is O(N) and doesn't scale.

Your primary technical challenge is to implement a scalable indexing algorithm.

## 4. Deliverables

Please provide the following at the end of the one-week period:
1. Source Code: Well-structured, commented code in a zipped file or a git
repository.
2. README.md: A comprehensive README file containing:
    ○ How to build and run your VecDB.
    ○ A clear API reference with code examples.
    ○ A brief "Design Doc" section explaining your architecture, data structures for HNSW, and persistence strategy.
    ○ A section on "Trade-offs & Future Improvements".
3. Tests: Include a set of basic unit and integration tests that verify the correctness of insert, search, and persistence operations.
4. (Optional but Recommended) Simple Benchmark: Report the results in your README.

## 5. Evaluation Criteria
    We will assess your submission based on:
    ● Correctness: Does the implementation work as expected? Are the search results accurate?
    ● Code Quality: Is the code clean, modular, readable, and well-documented?
    ● Algorithm Understanding: The quality of your implementation is a key differentiator. We want to see your thought process in the code and design doc.
    ● Architectural Design: How you separate concerns (e.g., indexing, storage, API).
    ● Technical Trade-offs: Your ability to recognize and articulate the limitations of your implementation and what you would improve given more time.

## 6. Getting Started & Resources

● HNSW Paper: The original paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" is the canonical source. Focus on the high-level idea.