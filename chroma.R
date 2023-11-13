

create_collection = function(path = "collection",
                             collection = "bmds") {
  chromadb = import("chromadb")
  chromadb$PersistentClient(path = path)$get_or_create_collection(collection)
}

get_collection = function(path = "collection",
                          collection = "bmds") {
  chromadb = import("chromadb")
  chromadb$PersistentClient(path = path)$get_collection(collection)
}

add_to_collection = function(collection, ids, documents,
                             embeddings = NULL, metadata = NULL) {
  collection$add(
    documents = documents,
    embeddings = embeddings,
    metadatas = metadata,
    ids = ids
  )
}

query_collection = function(query_embeddings, collection, n_results = 10L) {
  collection$query(
    query_embeddings = query_embeddings,
    n_results = n_results
  )
}
