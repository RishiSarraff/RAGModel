import pandas as pd
from embeddings import embeddingFunction, documentChunker, chunkWithMetadata, insertIntoChromaDB
from uuid import uuid4
from langchain.vectorstores import Chroma # we will use chroma as a way to centrally maintain all out text file embeddings into a non relational database

text1Df = pd.read_csv('textFiles/text1.txt', delimiter='\t', header=None, encoding='utf-8')
text2Df = pd.read_csv('textFiles/text2.txt', delimiter='\t', header=None, encoding='utf-8')
text3Df = pd.read_csv('textFiles/text3.txt', delimiter='\t', header=None, encoding='utf-8')

# print(text1Df)
# print(text2Df)
# print(text3Df)

# Article 1 = Renewing students’ motivation to learn through a Retreat Program
# Article 1 Length = 800
full_text1 = ' '.join(text1Df[0].astype(str))

# Article 2 = Light exercise can yield significant cognitive benefits, new research shows
# Article 2 Length = 589
full_text2 = ' '.join(text2Df[0].astype(str))

# Article 3 = Air pollution inside Philly’s subway is much worse than on the streets
# Article 3 Length = 669
full_text3 = ' '.join(text3Df[0].astype(str))


text1Chunks = documentChunker(full_text1)
text2Chunks = documentChunker(full_text2)
text3Chunks = documentChunker(full_text3)

text1FinalChunks = chunkWithMetadata(text1Chunks, text_source={"source":"https://www.teachermagazine.com/sea_en/articles/renewing-students-motivation-to-learn-through-a-retreat-program",
                                                  "article_title":"Renewing students’ motivation to learn through a Retreat Program",
                                                  "article_author":"Martha Goni",
                                                  "article_date":"September 28, 2022"})
text2FinalChunks = chunkWithMetadata(text2Chunks, text_source={"source":"https://theconversation.com/light-exercise-can-yield-significant-cognitive-benefits-new-research-shows-243559",
                                                  "article_title":"Light exercise can yield significant cognitive benefits, new research shows",
                                                  "article_author":"Jonathan G. Hakun",
                                                  "article_date":"November 27, 2024"})
text3FinalChunks = chunkWithMetadata(text3Chunks, text_source={"source":"https://theconversation.com/air-pollution-inside-phillys-subway-is-much-worse-than-on-the-streets-237843",
                                                  "article_title":"Air pollution inside Philly’s subway is much worse than on the streets",
                                                  "article_author":"Kabindra Shakya",
                                                  "article_date":"October 7, 2024"})

uuidsText1 = [str(uuid4()) for _ in range(len(text1FinalChunks))]
uuidsText2 = [str(uuid4()) for _ in range(len(text2FinalChunks))]
uuidsText3 = [str(uuid4()) for _ in range(len(text3FinalChunks))]

insertIntoChromaDB(text1FinalChunks, uuidsText1)
insertIntoChromaDB(text2FinalChunks, uuidsText2)
insertIntoChromaDB(text3FinalChunks, uuidsText3)

