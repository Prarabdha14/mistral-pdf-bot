from unstructured.partition.pdf import partition_pdf

# Path to your Transformer paper
pdf_path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf" 
# (Make sure this path matches where your file actually is!)

print("📖 Reading PDF with Unstructured... (This might take a moment)")

# The 'partition_pdf' function breaks the document into elements (Title, NarrativeText, etc.)
elements = partition_pdf(                                               
    filename=pdf_path,
    strategy="hi_res",           # uses layout detection (slower but smarter)
    infer_table_structure=True   # tries to extract tables
)

print(f"✅ Success! Extracted {len(elements)} elements.")

# Let's peek at what it found
print("\n--- First 5 Elements ---")
for el in elements[:5]:
    print(f"[{el.category}]: {el.text}")

print("\n--- Checking for Tables ---")
tables = [el for el in elements if el.category == "Table"]
if tables:
    print(f"Found {len(tables)} tables!")
    print("Example Table content:", tables[0].text[:100], "...")
else:
    print("No tables found.")