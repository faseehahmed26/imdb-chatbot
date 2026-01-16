import pandas as pd
import duckdb
import chromadb
from openai import OpenAI
from pathlib import Path
import sys
from tqdm import tqdm

from config import (
    CSV_PATH,
    DUCKDB_PATH,
    CHROMA_PATH,
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)


def clean_gross(gross_str):
    """Convert gross string like '28,341,469' to float"""
    if pd.isna(gross_str) or gross_str == '':
        return None
    try:
        # Remove commas and convert to float
        return float(str(gross_str).replace(',', ''))
    except:
        return None


def clean_runtime(runtime_str):
    """Convert runtime string like '142 min' to integer"""
    if pd.isna(runtime_str) or runtime_str == '':
        return None
    try:
        # Extract number from string like '142 min'
        return int(str(runtime_str).replace(' min', '').strip())
    except:
        return None


def clean_year(year_str):
    """Convert year string to integer"""
    if pd.isna(year_str) or year_str == '':
        return None
    try:
        # Handle cases like '(2019)' or '2019'
        year_clean = str(year_str).replace('(', '').replace(')', '').strip()
        return int(year_clean)
    except:
        return None


def clean_meta_score(score):
    """Convert meta score to float"""
    if pd.isna(score) or score == '':
        return None
    try:
        return float(score)
    except:
        return None


def clean_votes(votes_str):
    """Convert votes string to integer"""
    if pd.isna(votes_str) or votes_str == '':
        return None
    try:
        # Remove commas
        return int(str(votes_str).replace(',', ''))
    except:
        return None


def build_metadata_dict(row):
    """
    Build metadata dictionary excluding None values.
    ChromaDB doesn't accept None values, so we only include fields with actual data.
    """
    metadata = {}

    # Always include these core fields
    metadata['Series_Title'] = str(row['Series_Title'])
    metadata['Genre'] = str(row['Genre'])
    metadata['Director'] = str(row['Director'])
    metadata['Overview'] = str(row['Overview'])

    # Only include optional fields if they have values
    if pd.notna(row['Released_Year']):
        metadata['Released_Year'] = int(row['Released_Year'])

    if pd.notna(row['IMDB_Rating']):
        metadata['IMDB_Rating'] = float(row['IMDB_Rating'])

    if pd.notna(row['Meta_score']):
        metadata['Meta_score'] = float(row['Meta_score'])

    if pd.notna(row['Runtime']):
        metadata['Runtime'] = int(row['Runtime'])

    if pd.notna(row['Gross']):
        metadata['Gross'] = float(row['Gross'])

    if pd.notna(row['Star1']):
        metadata['Star1'] = str(row['Star1'])

    if pd.notna(row['Star2']):
        metadata['Star2'] = str(row['Star2'])

    if pd.notna(row['Star3']):
        metadata['Star3'] = str(row['Star3'])

    if pd.notna(row['Star4']):
        metadata['Star4'] = str(row['Star4'])

    if pd.notna(row['No_of_Votes']):
        metadata['No_of_Votes'] = int(row['No_of_Votes'])

    if pd.notna(row['Certificate']):
        metadata['Certificate'] = str(row['Certificate'])

    return metadata


def load_and_clean_data():
    """Load CSV and clean data"""
    print(f"Loading data from {CSV_PATH}...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} movies")

    # Clean data
    print("Cleaning data...")

    # Strip whitespace from string columns
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

    # Convert Gross to float
    df['Gross'] = df['Gross'].apply(clean_gross)

    # Convert Runtime to int
    df['Runtime'] = df['Runtime'].apply(clean_runtime)

    # Convert Released_Year to int
    df['Released_Year'] = df['Released_Year'].apply(clean_year)

    # Convert Meta_score to float
    df['Meta_score'] = df['Meta_score'].apply(clean_meta_score)

    # Convert No_of_Votes to int
    df['No_of_Votes'] = df['No_of_Votes'].apply(clean_votes)

    # Fill missing overviews with empty string
    df['Overview'] = df['Overview'].fillna('')

    # Convert IMDB_Rating to float (should already be, but ensure it)
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

    print("Data cleaning complete!")
    print(f"Data types:\n{df.dtypes}")

    return df


def setup_duckdb(df):
    """Create DuckDB database from dataframe"""
    print(f"\nSetting up DuckDB at {DUCKDB_PATH}...")

    # Remove existing database if it exists
    if DUCKDB_PATH.exists():
        print("Removing existing database...")
        DUCKDB_PATH.unlink()

    # Create connection
    con = duckdb.connect(str(DUCKDB_PATH))

    # Create table from dataframe
    print("Creating table...")
    con.execute("CREATE TABLE imdb AS SELECT * FROM df")

    # Add indexes for commonly queried columns
    print("Creating indexes...")
    try:
        con.execute("CREATE INDEX idx_year ON imdb(Released_Year)")
        con.execute("CREATE INDEX idx_rating ON imdb(IMDB_Rating)")
        con.execute("CREATE INDEX idx_director ON imdb(Director)")
    except Exception as e:
        print(f"Note: Could not create some indexes: {e}")

    # Verify
    count = con.execute("SELECT COUNT(*) FROM imdb").fetchone()[0]
    print(f"DuckDB setup complete! {count} movies in database")

    # Show sample
    print("\nSample data:")
    sample = con.execute(
        "SELECT Series_Title, Released_Year, IMDB_Rating, Genre FROM imdb LIMIT 3").fetchdf()
    print(sample)

    con.close()
    return True


def setup_chromadb(df):
    """Create ChromaDB vector store with embeddings"""
    print(f"\nSetting up ChromaDB at {CHROMA_PATH}...")

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Cannot create embeddings.")
        return False

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection("imdb_overviews")
        print("Deleted existing collection")
    except:
        pass

    # Create collection
    collection = chroma_client.create_collection(
        name="imdb_overviews",
        metadata={"description": "IMDB movie overviews and metadata"}
    )

    print("Generating embeddings...")

    # Prepare data for embedding
    batch_size = 100
    total_movies = len(df)

    for i in tqdm(range(0, total_movies, batch_size), desc="Embedding batches"):
        batch_df = df.iloc[i:i+batch_size]

        # Create texts for embedding
        texts = []
        metadatas = []
        ids = []

        for idx, row in batch_df.iterrows():
            # Combine title, genre, director, and overview for richer embeddings
            text = f"{row['Series_Title']} | {row['Genre']} | {row['Director']} | {row['Overview']}"
            texts.append(text)

            # Build metadata excluding None values (ChromaDB doesn't accept None)
            metadata = build_metadata_dict(row)
            metadatas.append(metadata)
            ids.append(f"movie_{idx}")

        # Generate embeddings using OpenAI
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]

            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"\nError processing batch {i//batch_size + 1}: {e}")
            continue

    # Verify
    count = collection.count()
    print(f"\nChromaDB setup complete! {count} movies indexed")

    # Test query
    print("\nTesting semantic search...")
    try:
        # Generate query embedding using OpenAI (same as document embeddings)
        query_text = "police detective crime investigation"
        query_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query_text]
        )
        query_embedding = query_response.data[0].embedding

        # Query using the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        print(
            f"Found {len(results['documents'][0])} results for '{query_text}'")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Safely access Released_Year which might not exist for some movies
            year = metadata.get('Released_Year', 'N/A')
            print(
                f"  {i+1}. {metadata['Series_Title']} ({year})")
    except Exception as e:
        print(f"Test query failed: {e}")
        print("Note: This doesn't affect the main embedding generation which completed successfully.")

    return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("IMDB Agent Data Setup")
    print("=" * 60)

    try:
        # Step 1: Load and clean data
        df = load_and_clean_data()

        # Step 2: Setup DuckDB
        setup_duckdb(df)

        # Step 3: Setup ChromaDB
        setup_chromadb(df)

        print("\n" + "=" * 60)
        print("Data setup complete!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  - Streamlit app: streamlit run src/app.py")
        print("  - Telegram bot: python -m src.telegram_bot")

    except Exception as e:
        print(f"\nERROR during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
