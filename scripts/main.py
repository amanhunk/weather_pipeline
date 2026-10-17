
import requests
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
POPULATION_FILE = PROJECT_DIR / "data" / "population.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "merged_weather.csv"

API_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(population_df):
    """Fetch current weather for each city's coordinates."""
    weather_records = []

    for _, city in population_df.iterrows():
        params = {
            "latitude": city["Latitude"],
            "longitude": city["Longitude"],
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        }

        try:
            response = requests.get(
                API_URL, params=params, timeout=30
            )
            response.raise_for_status()
            payload = response.json()
            current = payload["current"]

            weather_records.append({
                "City": city["City"],
                "Temperature_C": current.get("temperature_2m"),
                "Wind_Speed_kmh": current.get("wind_speed_10m"),
                "Observation_Time": current.get("time"),
            })

            print(f"Fetched: {city['City']}")

        except (requests.RequestException, KeyError, TypeError) as error:
            print(f"Failed: {city['City']} - {error}")

    return pd.DataFrame(
        weather_records,
        columns=[
            "City",
            "Temperature_C",
            "Wind_Speed_kmh",
            "Observation_Time",
        ],
    )


def main():
    # 1. Load the population and coordinate dataset
    if not POPULATION_FILE.exists():
        raise FileNotFoundError(
            f"Population file not found: {POPULATION_FILE}"
        )

    population_df = pd.read_csv(POPULATION_FILE)

    required_columns = {
        "City", "Country", "Population", "Latitude", "Longitude"
    }

    missing_columns = required_columns - set(population_df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {missing_columns}")

    # 2. Clean city names and enforce input data types
    population_df["City"] = population_df["City"].astype("string").str.strip()
    population_df["Country"] = (
        population_df["Country"].astype("string").str.strip()
    )

    population_df["Population"] = pd.to_numeric(
        population_df["Population"], errors="coerce"
    )
    population_df["Latitude"] = pd.to_numeric(
        population_df["Latitude"], errors="coerce"
    )
    population_df["Longitude"] = pd.to_numeric(
        population_df["Longitude"], errors="coerce"
    )

    # Remove records missing essential population or coordinate data
    population_df = population_df.dropna(
        subset=[
            "City", "Country", "Population", "Latitude", "Longitude"
        ]
    ).copy()

    population_df = population_df[
        population_df["Population"] >= 0
    ].copy()

    population_df["Population"] = (
        population_df["Population"].round().astype("int64")
    )

    # 3. Fetch weather data from Open-Meteo
    weather_df = fetch_weather(population_df)

    if weather_df.empty:
        raise RuntimeError(
            "No weather records were returned. Check your internet connection."
        )

    # 4. Merge using City because multiple cities can share a country
    merged_df = population_df.merge(
        weather_df,
        on="City",
        how="inner",
        validate="one_to_one",
    )

    # 5. Enforce numeric types
    merged_df["Population"] = merged_df["Population"].astype("int64")
    merged_df["Temperature_C"] = pd.to_numeric(
        merged_df["Temperature_C"], errors="coerce"
    ).astype("float64")
    merged_df["Wind_Speed_kmh"] = pd.to_numeric(
        merged_df["Wind_Speed_kmh"], errors="coerce"
    ).astype("float64")

    # 6. Handle missing weather values by removing incomplete rows
    merged_df = merged_df.dropna(
        subset=["Temperature_C", "Wind_Speed_kmh"]
    ).copy()

    # 7. Derive Fahrenheit and temperature categories
    merged_df["Temperature_F"] = (
        merged_df["Temperature_C"] * 9 / 5 + 32
    ).astype("float64")

    merged_df["Temperature_Category"] = pd.cut(
        merged_df["Temperature_C"],
        bins=[float("-inf"), 10, 25, float("inf")],
        labels=["Cold", "Mild", "Hot"],
        right=False,
    ).astype("string")

    # 8. Report missing values and data types
    print("\nMissing values in final dataset:")
    print(merged_df.isnull().sum())

    print("\nFinal column data types:")
    print(merged_df.dtypes)

    # 9. Save the processed dataset locally
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMerged weather and population data:")
    print(merged_df.head(10).to_string(index=False))

    print(f"\nPopulation records loaded: {len(population_df)}")
    print(f"Weather records returned: {len(weather_df)}")
    print(f"Final merged records: {len(merged_df)}")
    print(f"Saved output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
