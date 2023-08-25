import faiss

print("Hello faiss!")
index = faiss.index_factory(128, "Flat")
print(index)