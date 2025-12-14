import os
import asyncio
import openai
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
from typing import List
import json
from datetime import datetime, timedelta
# import nltk
# from nltk.stem import SnowballStemmer
# import chromadb

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
target_words = 5

target_language = "Spanish"
DICTIONARY_FILE = "word_dictionary.json"

# Download NLTK data for Spanish lemmatization
# try:
#     nltk.data.find('stemmers/snowball')
# except LookupError:
#     nltk.download('punkt', quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run news agent with configurable config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input.txt",
        help="Path to input file to spanify (default: input.txt)",
    )

    # parser.add_argument(  # TODO is not used now
    #     "--runs",
    #     type=int,
    #     default=None,
    #     help="How many times each agent should run (overrides the value in config file)",
    # )
    return parser.parse_args()


async def get_words_to_avoid() -> List[str]:
    """
    Read the dictionary file and return words that should be avoided based on usage rules:
    - 1 usage: no restriction
    - 2 usages: avoid for 3 days
    - 3 usages: avoid for 2 weeks
    - 4+ usages: avoid for 2 months
    """
    if not os.path.exists(DICTIONARY_FILE):
        return []
    
    with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    
    words_to_avoid = []
    current_date = datetime.now()
    
    for word, data in dictionary.items():
        usage_count = data.get("count", 0)
        last_used_str = data.get("last_used")
        
        if usage_count <= 1:
            continue
        
        if last_used_str:
            last_used = datetime.fromisoformat(last_used_str)
            
            if usage_count == 2:
                threshold = timedelta(days=3)
            elif usage_count == 3:
                threshold = timedelta(weeks=2)
            else:  # 4 or more
                threshold = timedelta(days=60)  # 2 months
            
            if current_date - last_used < threshold:
                words_to_avoid.append(word)
    
    return words_to_avoid

async def translate_news(curated_news_path: str, output_path: str, config: dict) -> str:
    """
    Process the text to replace several words in it with the target language,
    add IPA transliteration, and include explanations.
    """
    with open(curated_news_path, "r", encoding="utf-8") as f:
        curated_news = f.read()

    target_words = config.get("words", 5)

    prompt = f"""
    You are given a document. Replace several words in the document with their {target_language} translations, aim for {target_words} such words scattered across the entire document.
    
    REPLACEMENT RULES:
    #TODO
    1. Words should not be from the list of words to avoid.
    2. Words should be spread evenly across the document, not clustered together. Avoid replacing multiple words in the same sentence or adjacent sentences or a paragraph or a block of the text.
    3. In the source text replace exactly 10 different words with their {target_language} equivalents
    4. Vary the types of words replaced (nouns, verbs, adjectives, etc.)
    5. Use the most basic form: nominative/singular for nouns, infinitive for verbs
    6. Immediately after the {target_language} word, add its IPA transliteration in square brackets.
    7. Add a mini-glossary explaining the meaning of each replaced word in English. It includes:
        - First, the {target_language} word
        - Then, its IPA transliteration
        - then, a short English sentence defining the word without the word itself
        - Finally, the direct English translation as a single word
        e.g.:
        **Vocabulary:**
        - **[Gatto]** /[ˈɡat.to]/ - [A fluffy small predator which is popular as a pet. Cat.]
        - **[Correre]** /[korˈre.re]/ - [Moving fast with both feets in the air. To run.]
    8. Format the output as html file where the glossary is on the right in a separate box for the ease of reading. Make it visually appealing.
    ---
    
    SOURCE TEXT:
    {curated_news}

    List of words to avoid: {await get_words_to_avoid()}
    
    Provide the transformed text with {target_language} word replacements, IPA transliterations in square brackets immediately after each {target_language} word, and vocabulary explanations on the right side of the html file. Absolutely NONE additional commentaries, output should be ready to be written into .html.
    """

    response = await client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "medium"},
        input=prompt,
        max_output_tokens=40000,
        #TODO let it search transliteration rules if needed
    )

    spanified_news = response.output_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(spanified_news)

    return spanified_news


class ExtractWordsResponse(BaseModel):
    words: List[str]


async def extract_words(text: str) -> List[str]:
    prompt = f"""
    You are given a text where several words are in {target_language}.

    TASK:
    1. Extract all UNIQUE {target_language} words from the text.
    2. Return them as a JSON array of strings.

    TEXT:
    {text}

    """

    resp = await client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ExtractWordsResponse,
    )

    return resp.choices[0].message.parsed.words


#TODO find alternative method
# async def lemmatize_words(words: List[str]) -> List[str]:
#     """
#     Lemmatize Spanish words using NLTK's Snowball stemmer.
#     Returns the base form of each word.
#     """
#     stemmer = SnowballStemmer("spanish")
#     lemmatized = [stemmer.stem(word.lower()) for word in words]
#     return lemmatized

async def update_dictionary(words: List[str]) -> None:
    """
    Update the dictionary file with new word usage data.
    Increments usage count and updates last_used timestamp.
    """
    # Load existing dictionary or create new one
    if os.path.exists(DICTIONARY_FILE):
        with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
    else:
        dictionary = {}
    
    current_date = datetime.now().isoformat()
    
    # Update each word
    for word in words:
        if word in dictionary:
            dictionary[word]["count"] += 1
            dictionary[word]["last_used"] = current_date
        else:
            dictionary[word] = {
                "count": 1,
                "last_used": current_date
            }
    
    # Save updated dictionary
    with open(DICTIONARY_FILE, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


async def make_target_language_sentence(
    target_language_words: List[str], sentence_txt: str
) -> str:
    """,
    Generate the shortest meaningful foreign text using all of the given target language words,
    and save the sentence to a .txt file.
    """

    prompt = f"""
    You are given a few words from the {target_language} language. 

    TASKS:
    1. Using ALL of these words, create the SHORT {target_language} text
       that is still grammatical and makes sense.
    2. Avoid ":" and ";" in the sentence. Make it more like a person speaks.
    3. Return ONLY the text, without any additional commentary.


    INPUT:
    {target_language_words}
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "low"},
        input=prompt,
        max_output_tokens=4000,
    )

    result = response.output_text.strip()
    sentence = result.split("\n")[0].strip()

    # save sentence
    with open(sentence_txt, "w", encoding="utf-8") as f:
        f.write(sentence)

    return sentence


async def generate_audio(sentence: str, audio_path: str):
    """
    Generate speech audio for the given target language sentence.
    Saves the output to an MP3 file.
    """

    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=sentence,
    ) as response:
        response.stream_to_file(audio_path)

    return response


def print_section(message):
    """Print a message wrapped with separator lines."""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")


async def main(input_file: str = "input.txt"):
    print_section(f"Adding {target_language}")
    translated_news = await translate_news(input_file, "news_translated.html", config)
    # print(translated_news)

    print_section(f"Extracting {target_language} words...")
    target_language_words = await extract_words(translated_news)
    # print_section("Lemmatizing Spanish words...")
    # lemmatized_words = await lemmatize_words(target_language_words)
    # print(f"Lemmatized words: {lemmatized_words}")

    print_section("Updating word dictionary...")
    await update_dictionary(target_language_words)

    print_section("Creating Spanish vocabulary sentence...")
    sentence = await make_target_language_sentence(
        target_language_words, f"{target_language}_sentence.txt"
    )

    print_section("Generating audio...")
    await generate_audio(sentence, f"{target_language}_sentence.mp3")

    print(f"Sentence saved to {target_language}_sentence.txt")
    print(f"Audio saved to {target_language}_sentence.mp3")

    print("\nfinished!")


if __name__ == "__main__":
    args = parse_args()

    config_path = os.path.abspath(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_language = config.get("target_language", "Spanish")

    # cli_runs = args.runs
    # config_runs = config.get("runs", 2)
    # runs = cli_runs if cli_runs is not None else config_runs

    asyncio.run(main(args.input))
