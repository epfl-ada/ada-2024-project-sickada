import wikipediaapi
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_wikipedia(url):
    """
    Extract the main content text from a Wikipedia page given its URL.

    This function uses the `wikipediaapi` library to fetch the content of a Wikipedia page.
    It extracts the title from the provided URL, queries the Wikipedia API for the page,
    and retrieves the text if the page exists.

    Parameters:
    ----------
    url : str
        The URL of the Wikipedia page to extract text from.

    Returns:
    -------
    str
        The text content of the Wikipedia page. If the page does not exist, returns "Page not found."
    """
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='MyWikipediaApp/1.0 (https://example.com/; myemail@example.com)'
    )

    title = url.split("/")[-1].replace("_", " ")
    page = wiki.page(title)

    if not page.exists():
        return "Page not found."
    return page.text

def find_keywords(text, num_keywords=10):
    """
    Extract keywords from a text based on the frequency of nouns and proper nouns.

    This function uses the spaCy library to process the text, identifies nouns and proper nouns,
    and counts their occurrences. It then returns the most common keywords based on frequency.

    Parameters:
    ----------
    text : str
        The text from which to extract keywords.
    num_keywords : int, optional
        The number of keywords to return, based on frequency. Default is 10.

    Returns:
    -------
    list of str
        A list of the most common keywords in lowercase, limited to nouns and proper nouns.
    """

    doc = nlp(text)

    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    keyword_counts = Counter(keywords)

    common_keywords = [word for word, _ in keyword_counts.most_common(num_keywords)]

    return common_keywords

def meets_keyword_requirements(row, keywords, weights=None, min_occurrences=2, min_keywords=3):
    """
    Check if a row meets the specified keyword requirements based on weighted occurrences.

    This function calculates the weighted occurrences of each keyword in the `title`, `description`, 
    and `tags` fields of the provided row. It checks if the occurrences meet the specified 
    minimum threshold for each keyword and counts how many keywords satisfy this requirement.

    Parameters:
    ----------
    row : dict
        A dictionary representing a single row of data with `title`, `description`, and `tags` fields.
    keywords : list of str
        List of keywords to search for in the row.
    weights : dict, optional
        A dictionary with weights for `title`, `description`, and `tags`, 
        e.g., {'title': 2, 'description': 1, 'tags': 1.5}. If `None`, weights are set to 1 for all fields.
    min_occurrences : int, optional
        The minimum weighted occurrence threshold for a keyword to be considered as meeting the requirement. Default is 10.
    min_keywords : int, optional
        The minimum number of keywords that need to meet the `min_occurrences` threshold. Default is 3.

    Returns:
    -------
    bool
        Returns `True` if at least `min_keywords` of the keywords meet the weighted occurrence 
        threshold in the row. Otherwise, returns `False`.
    """
    
    if weights is None:
        weights = {'title': 1, 'description': 1, 'tags': 1}

    matching_keywords_count = 0

    for keyword in keywords:
        title_occurrences = row['title'].lower().count(keyword.lower()) * weights['title']
        description_occurrences = row['description'].lower().count(keyword.lower()) * weights['description']
        tags_occurrences = row['tags'].lower().count(keyword.lower()) * weights['tags']
        
        total_weighted_occurrences = title_occurrences + description_occurrences + tags_occurrences

        if total_weighted_occurrences >= min_occurrences:
            matching_keywords_count += 1

    return matching_keywords_count >= min_keywords
