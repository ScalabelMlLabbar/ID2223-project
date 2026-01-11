"""

Collect the top domains from hopswork and get the urls from their sitemap

"""

# imports
from src.phising_detection.utils import hopsworks_utils as hw
from src.phising_detection.data import sitemap_parser as parser
import pandas as pd
from datetime import datetime
import ast
import numpy as np




def collect_domains():
    """
    Read the dataframe with alla top 1M domains from a feature group in hopswork
    
    :return df: return the dataframe in feature group legit_domains
    """
    # connecting to hopswork
    project = hw.connect_to_hopsworks()

    df = hw.read_feature_group(
        project=project,
        feature_group_name="legit_domains",
        version=1,
        online=True
    )
    return df

def find_non_parsed_domains(all_domains):
    """
    Find the unique domains we haven't scanned for urls by comparing all domains to the current ones in a feature group
    
    :param all_domains: all domains from the latest top 1M trunco list

    return: The current domains in the feature group and the new unique ones
    """

    # connecting to hopswork
    project = hw.connect_to_hopsworks()

    current_domains = hw.read_feature_group(
        project=project,
        feature_group_name="scan_progress_legit_urls",
        version=4,
        online=False
    )
    
    # Assume df_small has 1000 domains
    small_set = set(current_domains["domain"])

    # Filter the large dataframe: keep only domains not in small_set
    unique_domains = all_domains[~all_domains["domain"].isin(small_set)]
    print("Number of unique domains: ", len(unique_domains))


    return current_domains, unique_domains


def get_urls_from_domain(domains, max_domains=50):
    """
    Extracts the urls from each domains site map and collect them into a dataframe
    
    :param domains: all potential domains to extract urls from.
    :param max_domains: max amount of domains to process URLs from. 

    :return df: dataframe with domains, their urls and a time of extraction
    """
    # Get the domains we want to process
    if max_domains < len(domains):
        domains_to_use = domains.head(max_domains)
    else:
        domains_to_use = domains
    
    domain_names = domains_to_use["domain"]

    # Retrive the urls
    rows = [] # a list of dicts

    for domain in domain_names:

        try:
            # Get time and date
            time_updated = datetime.now().isoformat()
            # retrive and parse the sitemap of the domain for urls
            urls = parser.get_urls_from_sitemap(domain=domain, max_urls=10)
            
            if len(urls) == 0:
                # if there are no urls, just add the domain
                rows.append({
                "domain": domain,
                "time_updated": time_updated,
                "urls": "['https://" + domain + "']"
                })
            else: 
                rows.append({
                "domain": domain,
                "time_updated": time_updated,
                "urls": str(urls)
                 })
                
        except Exception as e:
            print("Error while retrivning urls from sitemap: ", e)
        
    # Create dataframe 
    df = pd.DataFrame(rows)

    return df

def load_domainURLs_into_hopswork(new_domains, current_domains):
    """
    Combines the old and new domains and in a dataframe and uploads it to Hopswork

    :param new_domains: a pandas dataframe with a column for domain and another column with respective urls from the domains sitemap
    :param current_domains: the dataframe to append new domains and urls 
    """
    # Merge dataframes
    df_combined = pd.concat([new_domains, current_domains], ignore_index=True)
   
    # connecting to hopswork
    project = hw.connect_to_hopsworks()

    # uploading dataframe to feature group
    hw.upload_dataframe_to_feature_group(
        project=project,
        df=df_combined,
        feature_group_name="scan_progress_legit_urls",
        version=4,
        description="Scanned domains with timestamps and their respectively found URLs from the domain sitemap",
        primary_key="domain"
    )



def main():
    # Get all domains from hopswork
    domains = collect_domains()

    # Find all domains that haven't been parsed for URLs
    parsed_domains, non_parsed_domains = find_non_parsed_domains(domains)

    if len(non_parsed_domains) != 0:

        # Extract urls from the non parsed domains sitemap
        df_domain_urls = get_urls_from_domain(non_parsed_domains)

        # Load the new urls into hopswork
        load_domainURLs_into_hopswork(df_domain_urls, parsed_domains)
    
    else: 
        print("All domains in the latest Tranco list have already been processed")

if __name__ == "__main__":
    main()