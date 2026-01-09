"""Load the the gathered URLs from all domains stored in the feature group "scan_progress_legit_urls" individually into another feature group """

# imports 
from src.phising_detection.utils import hopsworks_utils as hw
import ast
import pandas as pd
import logging as log


def read_feature_group():
    """
    Read the feature group scan_progress_legit_urls from hopswork
    """
    # connecting to hopswork
    project = hw.connect_to_hopsworks()

    # read feature group
    df = hw.read_feature_group(
        project=project,
        feature_group_name="scan_progress_legit_urls",
        version=4,
        online=False
    )
    return df

def seperated_urls(df):
    """
    For each found urls from a domain. Extract each url individually and thereafter add all to new dataframe.
    All urls are individually checked that they can be decoded by utf-8

    :param df: Dataframe with a column "urls". The column contains string that looks like lists of urls
    """
    # holder for new dataframe 
    rows = []
    
    urls = df["urls"]

    # Extracting the urls from a string that looks like a list of urls 
    for url in urls:
        if url == "[]":
            continue
        else:
            try:
                # make the str that looks like a list into an actual list
                lst = ast.literal_eval(url)
            except (ValueError, SyntaxError):
                continue
            # Append each url to a row and set it to a non phishing url
            for element in lst:
                rows.append({
                "is_phishing": 0,
                "url": element
                 })

    # Create dataframe
    df_urls = pd.DataFrame(rows)
    # remove rows that have urls that can't be decoded by utf-8
    df_urls['url'] = df_urls['url'].apply(clean_url)
    # Create a unique id for each URL to use as primary key
    df_urls['id'] = range(1, len(df_urls) + 1)

    # debug: show URLs found
    log.info(f"Total URLs found: {len(df_urls)}")

    return df_urls


def clean_url(url):
    """
    Checking that all urls can de decoded with utf-8
    
    :param url: url in datatype str
    """
    if not isinstance(url, str):
        return url
    return url.encode('utf-8', 'ignore').decode('utf-8')

def load_into_hopswork(df):
    """
    Load dataframe into Hopsworks
    
    :param df: dataframe with individual urls
    """
    
    # connecting to hopswork
    project = hw.connect_to_hopsworks()

    # uploading dataframe to feature group
    hw.upload_dataframe_to_feature_group(
        project=project,
        df=df,
        feature_group_name="legit_urls_before_scan",
        version=2,
        description="Legitimate URLs formatted for phishing detection",
        primary_key=["id"]
    )

def main():
    # Get domains and their urls from hopswork
    domain_urls_df = read_feature_group()
    # seperate the urls from the domains and filter/process them
    seperated_urls_df = seperated_urls(domain_urls_df)
    # Load seperated urls into hopswork
    load_into_hopswork(seperated_urls_df)


if __name__ == "__main__":
    main()