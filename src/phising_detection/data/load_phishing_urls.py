"""

Collecting Phishing URLs from the Phishing database on Github and loading them into Hopswork

There are multiple pages on github the .txt files with active pshing urls. 
1. Gather the urls to all the .txt files witht the Phishing urls
2. Gather the data from the .txt files
3. Load the data into Hopswork

"""
# Imports
import requests
import pandas as pd
import sys
import os

# Add the src folder to sys.path so Python can see utils
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # go up one level from data/
sys.path.append(src_folder)

# Now import
from utils import hopsworks_utils as hw


# Global variables 
MANIFEST_LINK = "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/refs/heads/master/phishing-links-ACTIVE/phishing-links-ACTIVE.manifest.txt"
LINK_BASE = "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/refs/heads/master/phishing-links-ACTIVE/"

def request_manifest_link():
    
    """
    Requests and collects data from the manifest page.
    The manifest pages that holds what pages that contain the active Phishing urls
    
    """
    # Gather all links to the .txt files with the active Phising urls
    resp = requests.get(MANIFEST_LINK, timeout=10)
    resp.raise_for_status()
    content = resp.text
    txt_links = list(dict.fromkeys(line for line in content.split("\n") if line.strip()))
    
    return txt_links

def request_phishing_urls(links):
    """
    Requesting data from links to .txt files with phishing urls 
    
    :param links: a list with the suffix of the URLs for pages with active phishing links 
    :return all_dataframe: a dataframe containing the phishing URLs
    """
    all_urls = []

    for link_suffix in links: 
        # Request the data
        resp = requests.get(LINK_BASE + link_suffix, timeout=10)
        resp.raise_for_status()
        content = resp.text
        current_urls = list(dict.fromkeys(line for line in content.split("\n") if line.strip()))

        # Keep only http/https URLs
        current_urls = [url for url in current_urls if url.startswith(("http://", "https://"))]
        #print(f"amount of collected urls after filtering:{len(current_urls)}")

        # Append to all_urls
        all_urls += current_urls 
    
    # remove duplicates across files
    all_urls = list(dict.fromkeys(all_urls))


    # Create DataFrame
    df = pd.DataFrame({
    'url_id': range(len(all_urls)),
    'url': all_urls,
    'is_phishing': 1  # 0 for legitimate URLs
})
    
    return df

def load_into_hopswork(dataframe):
    """
    Loading the Phishing urls into Hopswork

    :param dataframes: A list of dataframes containing the Phishing URLs
    """
    # connecting to hopswork
    project = hw.connect_to_hopsworks()
    
    # load dataframe into a feature group
    fg =  hw.upload_dataframe_to_feature_group(
        project=project,
        df=dataframe,
        feature_group_name="phishing_urls",
        version=2,
        description="A dataframe with all the active Phishing URLs collected from the updated Phishing database",
        primary_key= ["url_id"],
        event_time= None,
        online_enabled=True
        )
    

def main():
    # Get the required link suffixs
    all_links = request_manifest_link()
    # Get the data from the links
    dataframes = request_phishing_urls(all_links)
    # load the data into Hopswork
    load_into_hopswork(dataframes)



if __name__ == "__main__":
    main()