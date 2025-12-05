import os
import asyncio
import openai
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
from typing import List
# import chromadb

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
number_of_news = 11

target_language = "Spanish"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run news agent with configurable config file"
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="config.yaml",
    #     help="Path to config YAML file (default: config.yaml)",
    # )

    parser.add_argument(
        "--skip_language",
        action="store_true",
        help="Skip spanification + Spanish sentence + audio generation (default: False)",
    )
    parser.add_argument(  # TODO is not used now
        "--runs",
        type=int,
        default=None,
        help="How many times each agent should run (overrides the value in config file)",
    )
    return parser.parse_args()


async def spanify_news(curated_news_path: str, output_path: str) -> str:
    """
    Process the text to replace several words in it with the target language,
    add IPA transliteration, and include explanations.
    """
    with open(curated_news_path, "r", encoding="utf-8") as f:
        curated_news = f.read()

    prompt = f"""
    You are given a document. Replace several words in the document with their {target_language} translations, aim for 10 such words scattered across the entire document.
    
    REPLACEMENT RULES:
    0. Words should be spread evenly across the document, not clustered together. Avoid replacing multiple words in the same sentence or adjacent sentences or a paragraph or a block of the text.
    1. In the source text replace exactly 10 words with their {target_language} equivalents
    2. Use the most basic form: nominative/singular for nouns, infinitive for verbs
    3. Immediately after the {target_language} word, add its IPA transliteration in square brackets
    4. Vary the types of words replaced (nouns, verbs, adjectives, etc.)

    
    In the end of the text add the vocabulary for the words that were just replaced. It should include the word, its IPA transliteration, English sentence defining the word and the English translation itself. E.G.
    **Vocabulary:**
    - **[Gatto]** /[ˈɡat.to]/ - [A fluffy, relatively small predator which is popular as a pet. Cat.]
    ---
    
    SOURCE TEXT:
    {curated_news}
    
    Provide the transformed text with {target_language} word replacements, IPA transliterations in square brackets immediately after each {target_language} word, and vocabulary explanations at hteb end of the text. Do NOT include any additional commentary.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "medium"},
        input=prompt,
        max_output_tokens=40000,
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


async def lemmatize_words(words: List[str]) -> List[str]:
    pass


async def make_spanish_sentence(
    target_language_words: List[str], sentence_txt: str
) -> str:
    """,
    Generate the shortest meaningful foreign text using all of the given target language words,
    and save the sentence to a .txt file.
    """

    prompt = f"""
    You are given a few words from the {target_language} language. 

    TASKS:
    2. Using ALL of these words, create the SHORT {target_language} text
       that is still grammatical and makes sense.
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


async def main():
    print_section("Spanifying news...")
    spanified_news = await spanify_news("news_curated.txt", "news_spanified.txt")
    # print(spanified_news)

    print_section("Extracting Spanish words...")
    target_language_words = await extract_words(spanified_news)

    print_section("Creating Spanish vocabulary sentence...")
    sentence = await make_spanish_sentence(
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

    cli_runs = args.runs
    config_runs = config.get("runs", 2)
    runs = cli_runs if cli_runs is not None else config_runs

    asyncio.run(main())
