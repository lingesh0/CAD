import openai

def normalize_labels(labels, api_key):
    """
    Normalize room labels using OpenAI API.

    Args:
        labels (list): List of raw labels to normalize.
        api_key (str): OpenAI API key.

    Returns:
        list: Normalized labels.
    """
    openai.api_key = api_key

    prompt = (
        "Normalize the following room labels to standard names:\n"
        f"{labels}\n"
        "Example: ['BED RM', 'KIT', 'LIV'] -> ['Bedroom', 'Kitchen', 'Living Room']"
    )

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.5
        )
        normalized = response.choices[0].text.strip().split("\n")
        return normalized
    except Exception as e:
        print("Error during label normalization:", e)
        return labels

if __name__ == "__main__":
    # Example usage
    raw_labels = ["BED RM", "KIT", "LIV"]
    api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key

    normalized_labels = normalize_labels(raw_labels, api_key)
    print("Normalized Labels:", normalized_labels)