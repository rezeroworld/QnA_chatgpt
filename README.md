# QnA_chatgpt

This chat bot is powered by ChatGPT language model. It uses external knowledge to answer questions about dental aligners. In this case, prior to the Q&A, a 300 pages knowledge book (Alpha Dentistry vol. 1- Digital Orthodontics INTERNATIONAL) was transformed into embeddings, and used to calculate a similarity metric against the presented query. Finally, the most relevent information is inserted into the prompt. The prompt is displayed with the bot answer to better understand where the answer comes from.

If you need more details about the processing of the book, you can check the text processing notebook.

## App Link

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatllama-japjh9chtd5ygyyneheu6r.streamlit.app/)