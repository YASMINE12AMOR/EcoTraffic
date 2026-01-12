from scripts.extract_weather import fetch_weather
from scripts.extract_pollution import fetch_pollution
from scripts.merge_and_load import merge_and_save

weather = fetch_weather()
pollution = fetch_pollution()

merge_and_save(weather, pollution)
print(weather.head())
print(pollution.head())
