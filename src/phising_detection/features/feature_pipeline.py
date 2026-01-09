""" Retrivning all necessary data from Tranco API's and Phishing database, processing them and storing them into the feature store in Hopswork before training"""

# Imports
from src.phising_detection.data.load_legit_domains import main as load_legit_domains
from src.phising_detection.data.load_phishing_urls import main as load_phishing_urls
from src.phising_detection.data.extract_urls import main as extract_urls
from src.phising_detection.data.seperate_domain_urls import main as seperate_domain_urls
import logging as log

def main():

    # Load all the latest active phishing urls into the feature store
    load_phishing_urls()
    log.info(f"Loading the latest active phishing urls completed!")

    # Load the latest top 1M domains from the Tranco list into the feature store
    load_legit_domains()
    log.info(f"Loading the latest top 1M domains completed!")

    # Find new domains that haven't been processed and extract urls from their sitemap, load domains and respective urls into the feature store 
    extract_urls()
    log.info(f"Extraction of urls from all new domains completed!")


    # Seperate the urls from the domains and filter bad urls, load them into the feature store
    seperate_domain_urls()
    log.info(f"Seperation of domains and url completed!")


if __name__ == "__main__":
    main()