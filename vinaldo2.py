# sudo apt install poppler-utils # to have pdftotext 
# pip install 'farm-haystack[faiss]'
# sudo apt-get install graphviz graphviz-dev
# pip install pygraphviz

# wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz &&
                   #tar -xvf xpdf-tools-linux-4.04.tar.gz && sudo cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

import logging
from haystack.nodes import DensePassageRetriever, FARMReader, PreProcessor, BM25Retriever, TextConverter
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, export_answers_to_csv
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
import pandas as pd
import argparse
import time
import os

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Retrieval')
#parser.add_argument('--outfile', type=str, default='answers/score2.txt', help='name of the deep model')
parser.add_argument('--overlap', type=int, default=2, help='overlap size of the sliding window')
args = parser.parse_args()

# Load questions
questions = None
# questions_df = pd.read_csv('data2/melville_answers.csv', header=0, on_bad_lines='skip')

# converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"]) # https://www.xpdfreader.com/
# converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
# all_docs = [converter.convert(file_path="data/Mardi.pdf", meta=None)[0]]
folder_list = ["1", "6", "8", "11", "15", "16"]
eval_start_time = time.time()
for folder in folder_list:
    if os.path.exists("./faiss_document_store.db"):
        os.remove("./faiss_document_store.db")       # This is needed for the next iteration
    path_folder = "data_txt/" + folder + "/"
    #print(path_folder)
    all_docs = convert_files_to_docs(dir_path=path_folder)
    #print("Folder path: ", path_folder)

# segment the questions (as done for the folders for better context/results)
    if folder == "1":
        print("In condition ", folder)
        questions = ["Why was he reconciled to Bartleby?", "Which church did he go to on Sunday?", "What was rolled away under his desk?", "What authors does he read?", "What is outside his office window?", "Where is he taken?"]
    elif folder == "6":
        print("In condition ", folder)
        questions = ["Why does he decide to desert the ship?", "Where is his companion from?", "Who does he kill?", "Who disappears?", "Who become his companions?", "Who do they leave behind?"]
    elif folder == "8":
        print("In condition ", folder)
        questions = ["Why does he go on a whaling voyage?", "Who does he share a bed with?", "Where is he when he loses his leg?", "What is behind the mask?", "What does the color white mean?", "What is the whale compared to?", "How is he saved?"]
    elif folder == "11":
        print("In condition ", folder)
        questions = ["What does he want to pawn?", "What is he called by the sailors?", "Who bullies him?", "What does he see in Launcelott’s-Hey?", "Who takes him to London?", "What happens to Miguel Saveda?"]
    elif folder == "15":
        print("In condition ", folder)
        questions = ["Where do they escape to?", "What afflicts him?", "Who lives in the dwelling-house?", "What happens at the Ti?", "What is at the silent spot?", "Why does he leave the island?"]
    elif folder == "16":
        print("In condition ", folder)
        questions = ["Who is his companion?", "What does he want to do to his jacket?", "Why is he called to the mast?", "What is fired out of the canon?", "How does he fall?", "What play is performed?"]

# Setting our parameters for the preprocessing
    # preprocessor = PreProcessor(
    #     clean_empty_lines=True,
    #     clean_whitespace=True,
    #     clean_header_footer=True,
    #     split_by="word",
    #     split_length=100,
    #     split_respect_sentence_boundary=True,
    #     split_overlap=args.overlap
    # )
# Actual preprocessing
    #preprocessed_docs = preprocessor.process(all_docs)

    document_store = InMemoryDocumentStore(use_bm25=True)

    files_to_index = [path_folder + "/" + f for f in os.listdir(path_folder)]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)

    # try:
    # # Instantiate the document store
    #     document_store = FAISSDocumentStore(faiss_index_factory_str="Flat") # (Facebook AI Similarity Search)
    # # Save all preprocessed documents to the document store
    #     document_store.write_documents(preprocessed_docs)
    # except ValueError:
    # # Reset document store, to make sure it is fine :)
    #     document_store.delete_documents()

    retriever = BM25Retriever(document_store=document_store)

    # retriever = DensePassageRetriever(
    #     document_store=document_store,
    #     query_embedding_model= "sentence-transformers/multi-qa-mpnet-base-dot-v1", #"facebook/dpr-question_encoder-single-nq-base", # change to MultiQA
    #     passage_embedding_model= "sentence-transformers/multi-qa-mpnet-base-dot-v1", # "facebook/dpr-ctx_encoder-single-nq-base", #
    #     max_seq_len_query=64,
    #     max_seq_len_passage=256,
    #     batch_size=16,
    #     use_gpu=True,
    #     embed_title=True,
    #     use_fast_tokenizers=True,
    # )
# We calculate the embeddings for all of our documents in the document store
    #document_store.update_embeddings(retriever)
    reader = FARMReader(
        model_name_or_path="deepset/roberta-base-squad2", use_gpu=True
        )

    pipe = ExtractiveQAPipeline(reader, retriever)
#pipe.draw() # prints our pipeline

# questions = questions_df.question.values.tolist()
# questions.append("Why was he reconciled to Bartleby?") # Add the bad row in the csv
#questions = ["Why was he reconciled to Bartleby?", "Which church did he go to on Sunday?", "What was rolled away under his desk?"]
#answers = []

    i = 0
    for question in questions:
        prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})
        #prediction2 = pipe.run(query=question)
        export_answers_to_csv(output_file="ans_bm25/r2_size" + str(args.overlap) + "folder_" + folder + "_ans" + str(i) + '.csv', agg_results=prediction)
        i += 1
    #answers.append(prediction)

eval_time_elapsed = time.time() - eval_start_time
print("Eval time: ", eval_time_elapsed)

# for answer in answers:
#     print("Q:", answer["query"])
#     print("A:", answer["answers"])
#     # print("score: ",answer["answers"][0]["score"], " probability: ",answer["answers"][0]["probability"])
#     print("\n")
# with open(args.outfile, "a+") as f:
#     s = f.write(f'Q: {answers} \n\n') #["query"]})  A: {answers["answers"]}' + "\n")
