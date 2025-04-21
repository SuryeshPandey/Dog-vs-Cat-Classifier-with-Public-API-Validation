import requests

# Function to classify breed using Wikipedia API
def classify_breed(breed_name: str) -> str:
    formatted_breed = breed_name.replace(' ', '_')
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_breed}"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        extract = data.get("extract", "").lower()
        if "dog" in extract:
            return "dog"
        elif "cat" in extract:
            return "cat"
    return "not found"

# Function to get breed from TheDogAPI if Wikipedia fails
def get_dog_breed_from_api(breed_name: str, api_key: str) -> str:
    url = "https://api.thedogapi.com/v1/breeds"
    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        breeds = response.json()
        breed_name_lower = breed_name.lower()
        for breed in breeds:
            if breed_name_lower in breed["name"].lower():
                return "dog"
    return "cat"  # fallback to cat if no match found in dog list
