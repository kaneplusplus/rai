library(purrr)
library(readr)
library(S7)
library(reticulate)
library(foreach)
library(progress)
library(itertools)

class_llm = new_class("class_llm", abstract = TRUE)

embed = new_generic("embed", c("text", "llm"))

complete = new_generic("complete", c("text", "llm"))

answer = new_generic("complete", c("question", "llm"))

class_huggingface_llm = new_class(
  name = "class_huggingface_llm",
  parent= class_llm,
  properties = list(
    model = class_any,
    tokenizer = class_any
  )
)

huggingface_llm = function(tokenizer, model) {
  #from transformers import AutoModelForCausalLM, AutoTokenizer
  transformers = import("transformers")
  builtins = import_builtins()
  torch = import("torch")
  builtins$setattr(
    torch$distributed, "is_initialized", py_eval("lambda : False")
  )
  tokenizer = transformers$AutoTokenizer$from_pretrained(tokenizer)
  model = transformers$AutoModelForCausalLM$from_pretrained(model)
  if (is.null(tokenizer$pad_token)) { 
    tokenizer$add_special_tokens(py_dict("pad_token", "[PAD]"))
    model$resize_token_embeddings(py_len(tokenizer))
  }
  class_huggingface_llm (
    model = model,
    tokenizer = tokenizer
  )  
}

mean_pooling = function(model_output, attention_mask) {
  #First element of model_output contains all token embeddings
  torch = import("torch")
  token_embeddings = model_output[1]
  input_mask_expanded = attention_mask$unsqueeze(-1L)$expand(token_embeddings$logits$size())$float()
  sum_embeddings = torch$sum(torch$multiply(token_embeddings$logits, input_mask_expanded), 1L)
  sum_mask = torch$clamp(input_mask_expanded$sum(1L), min=1e-9)
  ret = purrr::reduce(sum_embeddings$div(sum_mask)$tolist(), rbind)
  if (!is.matrix(ret)) {
    ret = matrix(ret, nrow = 1)
  }
  rownames(ret) = as.character(seq_len(nrow(ret)))
  ret
}

method(embed, list(class_character, class_huggingface_llm)) = 
  function(text, llm, ...) {

  encoded_input = llm@tokenizer(
    text,
    padding = TRUE,
    truncation = TRUE,
    max_length = 1024L,
    return_tensors = 'pt'
  )
  model_output = llm@model(
    input_ids = encoded_input$input_ids,
    attention_mask = encoded_input$attention_mask
  )
  mean_pooling(model_output, encoded_input$attention_mask) |>
    (\(x) {rownames(x) = text; x})()
}

class_llama = new_class(
  name = "class_llama", 
  parent = class_llm,
  properties = list(
    handle = class_any
  )
)

import_llama = function(model_path, n_ctx = 4096L, n_gpu_layers = 1L, 
                        use_mmap = TRUE, use_mlock = TRUE, embedding = TRUE,
                        verbose = FALSE) {

  if (!is.character(model_path) || length(model_path) != 1) {
    stop("The path argument should be a single string path to the model.")
  }

  llama = import("llama_cpp")
  llama$Llama(
    model_path = model_path,
    n_ctx = n_ctx,
    n_gpu_layers = n_gpu_layers,
    use_mmap = use_mmap,
    use_mlock = use_mlock,
    embedding = embedding,
    verbose = verbose
  ) |> class_llama()
}

method(embed, list(class_character, class_llama)) = 
  function(text, llm, ..., max_tokens = 4096L, temperature = 0., 
           as_matrix = FALSE, progress = interactive(),
           chunks = 20) {
  if (progress) {
    pb = progress_bar$new(total = chunks)
  }
  ret = foreach(tc = isplitVector(text, chunks = chunks)) %do% {
    if (progress) {
      pb$tick()
    }
    map(llm@handle$create_embedding(tc)$data, ~ .x[[2]])
  }
  ret = unlist(ret, recursive = FALSE)
  if (as_matrix) {
    reduce(ret, rbind)
  } else {
    ret
  }  
}

method(complete, list(class_character, class_llama)) = 
  function(text, llm, max_tokens = 2048L) {
    llm@handle(text, max_tokens = max_tokens) 
}

method(answer, list(class_character, class_llama)) = 
  function(question, llm, ...) {
    llm@handle(paste0("Q: ", question, " A:"), max_tokens=2048, 
      stop=c("Q:", "\n"))
}
