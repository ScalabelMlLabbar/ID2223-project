"""
Collecting the top 1 000 000 domains on the internet from the tranco list (https://tranco-list.eu/) and loading them into Hopswork.

"""

#imports
import requests
import pandas as pd
import sys
import os
from src.phising_detection.utils import hopsworks_utils as hw



# Global Variables 
URL = "https://tranco-list.eu/api/lists/date/latest"



def request_domain_list():
    """
    Requests the latest tranco list available through API

    return: reponse.json(): Returns the response from the API call
    """

    # API URL
    response = requests.get(URL)

    if response.status_code == 200:
        return response.json()
    else:
        print("Something went wrong when fetching URL...")
        response.raise_for_status()

def download_data(data):
    """
    Docstring for download_data
    
    :param data: Description
    """
    # Get the download link from the dict
    download_link = data['download']

    # Evaluate the response
    response = requests.get(download_link)

    if response.status_code == 200:
        # format the data
        return response.text

    else:
        print("Something went wrong when trying to download data from URL...")
        response.raise_for_status()


def load_into_hopswork(domain_list):
    """
    Docstring for load_into_hopswork
    
    :param domain_list: A list with all 1 000 000 top domains
    """
    # Formatting
    domain_dict = {}

    lines = domain_list.splitlines()

    for line in lines:
        if not line:
            continue  # skip empty lines

        rank, domain = line.split(",", 1)
        domain_dict[domain] = int(rank)

    # Create Dataframe
    df = pd.DataFrame.from_dict(
        domain_dict,
        orient="index",
        columns=["rank"]
        ).reset_index()

    df.columns = ["domain", "rank"]

    # Load into hopswork
    
    # connecting to hopswork
    project = hw.connect_to_hopsworks()
    
    # load dataframe into a feature group
    fg =  hw.upload_dataframe_to_feature_group(
        project=project,
        df=df,
        feature_group_name="legit_domains",
        version=1,
        description="A dataframe with all top 1 000 000 domains and their rank from the tranco list",
        primary_key= ["domain"],
        event_time= None,
        online_enabled=True
        )
    

    
def main():
    # Call the tranco API to get the link to download the list
    response = request_domain_list()
    # Download from the link and gather the data
    domain_list = download_data(response)
    # Load into hopswork
    load_into_hopswork(domain_list)


if __name__ == "__main__":
    main()