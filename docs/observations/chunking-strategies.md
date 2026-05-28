# Observations

# Corpus
Flattening the corpus directory caused silent overwrites on name collision, this negatively affected the identification of source files in queries.json. Re-ingested with tree preserved.

# Tail for chunk_fixed
The last chunk almost never divides evenly. When the window runs past the end of the text, we take what's left. But how short is too short? If the final chunk is 12 characters, is that a real chunk or noise? We decided to make this length a tuneable function parameter keeping a default value of 25 for now.

# Chunkers
We noticed that although the primary function of a chunker is to just split the string / doc basis the defined strategy they still have a dependecy on origin of the document (doc_id & source_file)

chunk_recursive does not apply overlap between separator-split chunks; overlap only applies at the character-split base case. This is deemed acceptable because for this chunker, the separators preserve semantic boundaries. This means chunk_recursive performs better on well-structured documents where paragraph boundaries coincide with topic boundaries, and may underperform on dense prose where context bleeds across paragraphs."
