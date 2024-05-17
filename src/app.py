# ruff: noqa: E402
import streamlit as st

st.set_page_config(
    page_title="NLP Web App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://www.github.com/abhiyantaabhishek",
        "Report a bug": "https://www.github.com/abhiyantaabhishek",
        "About": "This is a simple NLP web app made with streamlit",
    },
)

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from wordcloud import WordCloud

# NLP Pkgs
from deep_translator import GoogleTranslator
from textblob import TextBlob
import neattext as nt
import spacy

from collections import Counter
import re


# Summarization Function
def summarize_text(text, num_sentences=3) -> str:
    # Remove special characters and convert text to lowercase
    clean_text = re.sub("[^a-zA-Z]", " ", text).lower()

    # Split the text into words
    words = clean_text.split()

    # Calculate the frequency of each word
    word_freq = Counter(words)

    # Sort the words based on their frequency in descending order
    sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)

    # Exract the top `num_sentences` most frequent words
    top_words = sorted_words[:num_sentences]

    # Create a summary by joining the top words
    summary = " ".join(top_words)

    return summary


@st.cache_data
# Lemma and Tokens Function
def text_analyzer(text) -> list:
    # import English library
    nlp = spacy.load("en_core_web_sm")

    # Create an NLP object
    doc = nlp(text)

    # Extract tokens and lemma
    all_data = [
        ('"Token":{}, "Lemma":{}'.format(token.text, token.lemma_)) for token in doc
    ]

    return all_data


def main():
    """NLP web app with Streamlit"""

    title_template = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">NLP Web App</h1>
    </div>
    """
    st.markdown(title_template, unsafe_allow_html=True)

    subheader_template = """
    <div style="background-color:#464e5f;padding:5px;margin:2px;border-radius:10px">
    <h3 style="color:white;text-align:center;">Powered by Streamlit</h3>
    </div>
    """
    st.markdown(subheader_template, unsafe_allow_html=True)
    # st.sidebar.image("download.jpeg", use_column_width=True)

    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Text Analysis":
        st.subheader("Text Analysis")
        st.write("")

        raw_text = st.text_area(
            "Write something", "Enter a text in English...", height=350
        )

        if st.button("Analyze"):
            if len(raw_text) == 0:
                st.warning("Enter some text...")
            else:
                st.info("Basic Functions")

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Basic Info"):
                        st.info("Text Stats")
                        word_desc = nt.TextFrame(raw_text).word_stats()
                        result_desc = {
                            "Length of Text": word_desc["Length of Text"],
                            "Num of Vowels": word_desc["Num of Vowels"],
                            "Num of Consonants": word_desc["Num of Consonants"],
                            "Num of Stopwords": word_desc["Num of Stopwords"],
                        }
                        st.write(result_desc)

                    with st.expander("Stopwords"):
                        st.success("Stop Words List")
                        stopwords = nt.TextExtractor(raw_text).extract_stopwords()
                        st.error(stopwords)

                with col2:
                    with st.expander("Processed Text"):
                        st.success("Stopwords Excluded Text")
                        processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                        st.write(processed_text)

                    with st.expander("Plot Wordcloud"):
                        st.success("Wordcound")
                        wordcloud = WordCloud().generate(processed_text)
                        fig = plt.figure(1, figsize=(20, 10))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(fig)

                st.write("")
                st.write("")
                st.info("Advanced Features")

                col3, col4 = st.columns(2)

                with col3:
                    with st.expander("Tokens and Lemma"):
                        st.write("Tokens and Lemma")
                        processed_text_mid = processed_text
                        processed_text_mid = str(
                            nt.TextFrame(processed_text_mid).remove_puncts()
                        )
                        processed_text_fin = str(
                            nt.TextFrame(processed_text_mid).remove_special_characters()
                        )
                        tandl = text_analyzer(processed_text_fin)
                        st.json(tandl)

                with col4:
                    with st.expander("Text Summarization"):
                        st.write("Text Summarization")
                        summary_result = summarize_text(raw_text)
                        st.write(summary_result)

    if choice == "Translation":
        st.subheader("Translation")
        st.write("")
        st.write("")
        raw_text = st.text_area(
            "Original Text", "Write something here to be translated", height=200
        )
        if len(raw_text) < 3:
            st.warning("Enter provide text with at least 3 characters")
        else:
            target_lang = st.selectbox("Target Language", ["German", "French", "Spanish", "Italian", "Polish"])
            if target_lang == "German":
                target_lang = "de"
            elif target_lang == "French":
                target_lang = "fr"
            elif target_lang == "Polish":
                target_lang = "pl"
            else:
                target_lang = "it"

        if st.button("Translate"):
            with st.spinner("Translating..."):
                translator = GoogleTranslator(source="auto", target=target_lang)
                translated_text = translator.translate(raw_text)
                st.write(translated_text)


    if choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        st.write("")
        st.write("")
        raw_text = st.text_area(
            "Text to analyze", "Enter text here...", height=200
        )
        if st.button("Analyze"):
            blob = TextBlob(raw_text)
            if len(raw_text) == 0:
                st.warning("Enter some text...")
            else:
                blob = TextBlob(raw_text)
                st.info("Sentiment Analysis")
                st.write(blob.sentiment)
                st.write("")

    if choice == "About":
        st.subheader("About")
        st.write("")
        st.markdown("""
        ### NLP Web App made with streamlit

        for info:
        - [streamlit](https://streamlit.io/)
        """)


if __name__ == "__main__":
    main()
